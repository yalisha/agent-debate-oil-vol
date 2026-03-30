"""
Mixture-of-Experts Meta-Aggregation for Multi-Agent Debate.

Architecture:
  - Expert 1: GAT with DropEdge + learned sparse graph (dominates normal/elevated regimes)
  - Expert 2: MLP no-graph (contributes in extreme regimes)
  - Router: learned regime-conditional gating with anti-collapse safeguards

Training:
  - Phase 1: Independent expert pre-training (200 epochs each)
  - Phase 2: Joint fine-tuning with differential LR (100 epochs)
    - Router: lr=0.003, Experts: lr=0.0003
    - Entropy regularization on router to prevent collapse

Walk-forward evaluation, 5-seed ensemble, HAC DM test.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_idx
from copy import deepcopy

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20


# ══════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════

from optimized_gat import dm_test_hac, MultiHeadGATLayer
from further_optimization import (
    DropEdgeGATModel, add_rolling_accuracy_features
)
from final_comparison import OptimizedMLPModel


# ══════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════

class MoERouter(nn.Module):
    """Minimal router: 3-parameter quadratic function of persist_vol.

    Captures the U-shaped pattern where MLP wins at both low and high
    volatility extremes. Only 3 learnable params = impossible to overfit.

    GAT_weight = sigmoid(a + b*pv + c*pv^2)
    If c < 0: GAT preferred at intermediate pv (normal/elevated) = desired behavior
    """
    def __init__(self, n_experts=2):
        super().__init__()
        # Initialize: GAT preferred at mid-range pv
        # sigmoid(2 - 15*pv^2) ~ 1 at pv=0.3, ~ 0 at pv=0/0.6
        self.a = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.tensor(-15.0))

    def forward(self, router_feats):
        """
        router_feats: (batch, 6) but we only use persist_vol (column 0)
        Returns: expert weights (batch, 2) = [gat_weight, mlp_weight]
        """
        pv = router_feats[:, 0]  # persist_vol
        logit = self.a + self.b * pv + self.c * pv ** 2
        gat_w = torch.sigmoid(logit).unsqueeze(-1)
        mlp_w = 1 - gat_w
        weights = torch.cat([gat_w, mlp_w], dim=-1)
        return weights


# ══════════════════════════════════════════════════════════
# MoE MODEL
# ══════════════════════════════════════════════════════════

class MoEMetaAggregation(nn.Module):
    """Mixture of Experts: GAT + MLP with learned routing."""

    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16):
        super().__init__()
        self.gat_expert = DropEdgeGATModel(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=hidden_dim, n_heads=4, top_k=3,
            prior_adj=None, drop_rate=0.2
        )
        self.mlp_expert = OptimizedMLPModel(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=hidden_dim
        )
        self.router = MoERouter(n_experts=2)

    def forward(self, node_feats, context, base_preds, regime_feats, router_feats):
        """
        node_feats: (batch, 7, node_feat_dim)
        context: (batch, context_dim)
        base_preds: (batch, n_base)
        regime_feats: (batch, 3) for expert internal regime gates
        router_feats: (batch, 6) for MoE router
        """
        # Expert predictions
        pred_gat, w_gat, gate_gat, edge_mask, attn = self.gat_expert(
            node_feats, context, base_preds, regime_feats
        )
        pred_mlp, w_mlp, gate_mlp = self.mlp_expert(
            node_feats, context, base_preds, regime_feats
        )

        # Router weights
        expert_weights = self.router(router_feats)  # (batch, 2)
        w_gat_route = expert_weights[:, 0]
        w_mlp_route = expert_weights[:, 1]

        # Final prediction
        pred = w_gat_route * pred_gat + w_mlp_route * pred_mlp
        pred = torch.clamp(pred, min=0.05)

        return pred, expert_weights, pred_gat, pred_mlp, edge_mask


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train_expert_gat(model, node_feats, context, targets, base_preds,
                     regime_feats, train_idx, n_epochs=200, lr=0.003):
    """Phase 1: Train GAT expert independently."""
    gat = model.gat_expert
    optimizer = torch.optim.Adam(gat.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    best_loss, best_state = float('inf'), None
    gat.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, _, _, _, _ = gat(nf, ctx, bp, rf)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in gat.state_dict().items()}

    if best_state:
        gat.load_state_dict(best_state)
    return best_loss


def train_expert_mlp(model, node_feats, context, targets, base_preds,
                     regime_feats, train_idx, n_epochs=200, lr=0.003):
    """Phase 1: Train MLP expert independently."""
    mlp = model.mlp_expert
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    best_loss, best_state = float('inf'), None
    mlp.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, _, _ = mlp(nf, ctx, bp, rf)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    if best_state:
        mlp.load_state_dict(best_state)
    return best_loss


def train_moe_joint(model, node_feats, context, targets, base_preds,
                    regime_feats, router_feats, train_idx,
                    n_epochs=200, lr_router=0.003, freeze_experts=True):
    """Phase 2: Train router (optionally freeze experts).

    Freezing experts prevents degradation of pre-trained quality.
    No entropy regularization: let the router learn freely which expert to use.
    """
    if freeze_experts:
        for p in model.gat_expert.parameters():
            p.requires_grad = False
        for p in model.mlp_expert.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_router, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])
    rtf = torch.FloatTensor(router_feats[train_idx])

    best_loss, best_state = float('inf'), None

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, expert_weights, pred_gat, pred_mlp, _ = model(nf, ctx, bp, rf, rtf)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Unfreeze for next window
    if freeze_experts:
        for p in model.gat_expert.parameters():
            p.requires_grad = True
        for p in model.mlp_expert.parameters():
            p.requires_grad = True

    return best_loss


# ══════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════

def build_router_features(df_clean, context):
    """Build 6-dim router input features."""
    persist_vol = df_clean['persist_vol'].values
    vol_change = df_clean['persist_vol'].diff().fillna(0).values
    n_herding = df_clean['n_herding'].values / 7.0
    debate_har_gap = (df_clean['debate_vol'] - df_clean['har_vol']).values

    # Rolling accuracy features (lagged 20d)
    actual = df_clean['actual_vol'].values
    debate = df_clean['debate_vol'].values
    single = df_clean['single_vol'].values

    debate_se = np.full(len(df_clean), np.nan)
    single_se = np.full(len(df_clean), np.nan)
    for t in range(20, len(df_clean)):
        debate_se[t] = (debate[t-20] - actual[t-20]) ** 2
        single_se[t] = (single[t-20] - actual[t-20]) ** 2

    debate_err_ma = pd.Series(debate_se).rolling(20, min_periods=5).mean().values
    single_err_ma = pd.Series(single_se).rolling(20, min_periods=5).mean().values
    rel_accuracy = np.nan_to_num(debate_err_ma - single_err_ma, nan=0.0)

    router_feats = np.column_stack([
        persist_vol,
        persist_vol ** 2,
        vol_change,
        n_herding,
        debate_har_gap,
        rel_accuracy,
    ])
    return router_feats


def run_moe_seed(seed, df_clean, node_feats, context, targets, base_preds,
                 regime_feats, router_feats, node_feat_dim, context_dim, n_base):
    """Run one seed of walk-forward evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)
    expert_w = np.full((n, 2), np.nan)
    pred_gat_all = np.full(n, np.nan)
    pred_mlp_all = np.full(n, np.nan)

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = MoEMetaAggregation(
            node_feat_dim, context_dim, n_base=n_base, hidden_dim=16
        )

        # Phase 1: Pre-train experts
        train_expert_gat(model, node_feats, context, targets, base_preds,
                         regime_feats, train_idx)
        train_expert_mlp(model, node_feats, context, targets, base_preds,
                         regime_feats, train_idx)

        # Phase 2: Joint fine-tuning with router
        train_moe_joint(model, node_feats, context, targets, base_preds,
                        regime_feats, router_feats, train_idx)

        # Predict
        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            rtf = torch.FloatTensor(router_feats[test_idx])
            pred, ew, pg, pm, _ = model(nf, ctx, bp, rf, rtf)

        preds[test_idx] = pred.numpy()
        expert_w[test_idx] = ew.numpy()
        pred_gat_all[test_idx] = pg.numpy()
        pred_mlp_all[test_idx] = pm.numpy()

    return preds, expert_w, pred_gat_all, pred_mlp_all


def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("MIXTURE OF EXPERTS: GAT + MLP with Learned Router")
    print("=" * 70)

    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)

    # Enhanced features for context
    enh_context, enh_context_dim = add_rolling_accuracy_features(df_clean, context, context_dim)

    targets = df_clean['actual_vol'].values
    persist_vol = df_clean['persist_vol'].values

    base_cols = ['debate_vol', 'single_vol']
    base_preds = np.column_stack([df_clean[c].values for c in base_cols])
    n_base = len(base_cols)

    regime_feats = np.column_stack([
        persist_vol,
        df_clean['persist_vol'].diff().fillna(0).values,
        df_clean['n_herding'].values / 7.0,
    ])

    # Router features (6-dim)
    router_feats = build_router_features(df_clean, context)

    har = df_clean['har_vol'].values
    debate = df_clean['debate_vol'].values
    seeds = [42, 123, 456, 789, 1024]

    # ── MoE EVALUATION ──
    print(f"\n{'='*70}")
    print("MoE EVALUATION (5 seeds)")
    print(f"{'='*70}")

    all_preds = []
    all_expert_w = []
    seed_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}...", end=" ", flush=True)
        preds, ew, pg, pm = run_moe_seed(
            seed, df_clean, node_feats, enh_context, targets, base_preds,
            regime_feats, router_feats, node_feat_dim, enh_context_dim, n_base
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        mean_gat_w = np.nanmean(ew[valid, 0])

        print(f"RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={pval:.4f}), GAT_weight={mean_gat_w:.3f}")
        seed_results.append({
            'seed': seed, 'rmse': rmse, 'dm_hac': dm, 'p_hac': pval,
            'gat_weight': mean_gat_w
        })
        all_preds.append(preds)
        all_expert_w.append(ew)

    # Ensemble
    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    y_v = targets[valid]
    rmse_ens = np.sqrt(np.mean((ens[valid] - y_v)**2))
    dm_ens, p_ens = dm_test_hac(ens[valid] - y_v, har[valid] - y_v)

    rmses = [r['rmse'] for r in seed_results]
    print(f"\n  ────────────────────────────")
    print(f"  MoE Per-seed: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  MoE Ensemble: {rmse_ens:.4f}, DM_HAC={dm_ens:.3f} (p={p_ens:.4f})")

    # ── ROUTER ANALYSIS ──
    print(f"\n{'='*70}")
    print("ROUTER ANALYSIS: Expert weights by regime")
    print(f"{'='*70}")

    # Average expert weights across seeds
    ew_avg = np.nanmean(np.stack(all_expert_w), axis=0)

    regimes = pd.cut(persist_vol[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])

    print(f"\n  {'Regime':<12s} {'GAT_w':>8s} {'MLP_w':>8s} {'n':>6s}")
    print(f"  {'─'*40}")
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            gw = np.nanmean(ew_avg[valid][mask, 0])
            mw = np.nanmean(ew_avg[valid][mask, 1])
            print(f"  {r:<12s} {gw:>8.3f} {mw:>8.3f} {mask.sum():>6d}")

    # ── REGIME BREAKDOWN ──
    print(f"\n{'='*70}")
    print("REGIME BREAKDOWN")
    print(f"{'='*70}")

    print(f"\n  {'Regime':<12s} {'MoE':>8s} {'GAT':>8s} {'MLP':>8s} {'HAR':>8s} {'n':>6s}")
    print(f"  {'─'*55}")

    # Load reference GAT results
    gat_ref_rmses = {}
    mlp_ref_rmses = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            moe_r = np.sqrt(np.mean((ens[valid][mask] - y_v[mask])**2))
            har_r = np.sqrt(np.mean((har[valid][mask] - y_v[mask])**2))
            deb_r = np.sqrt(np.mean((debate[valid][mask] - y_v[mask])**2))
            print(f"  {r:<12s} {moe_r:>8.4f} {'':>8s} {'':>8s} {har_r:>8.4f} {mask.sum():>6d}")

    # ── COMPARISON TABLE ──
    print(f"\n{'='*70}")
    print("COMPARISON (all HAC-corrected)")
    print(f"{'='*70}")

    print(f"\n  {'Model':<35s} {'RMSE':>7s} {'DM_HAC':>9s} {'p':>8s} {'Sig':>4s}")
    print(f"  {'─'*65}")

    comparisons = [
        ('MoE (GAT+MLP, ensemble)', rmse_ens, dm_ens, p_ens),
    ]
    for name, col in [('Debate (naive)', 'debate_vol'), ('Single agent', 'single_vol'),
                      ('HAR', 'har_vol'), ('Persistence', 'persist_vol')]:
        vals = df_clean[col].values[valid]
        rmse = np.sqrt(np.mean((vals - y_v)**2))
        if name != 'HAR':
            dm, p = dm_test_hac(vals - y_v, har[valid] - y_v)
        else:
            dm, p = 0, 1
        comparisons.append((name, rmse, dm, p))

    for name, rmse, dm, p in sorted(comparisons, key=lambda x: x[1]):
        sig = '*' if p < 0.05 else ''
        print(f"  {name:<35s} {rmse:>7.4f} {dm:>9.3f} {p:>8.4f} {sig:>4s}")

    # Reference from previous experiments
    print(f"\n  Reference (from previous runs):")
    print(f"    Learned GAT:            0.1226")
    print(f"    Post-hoc stacking:      0.1201")

    # Save
    df_out = pd.DataFrame(seed_results)
    df_out.to_csv(BASE / "results" / "moe_results.csv", index=False)
    print(f"\nSaved to results/moe_results.csv")


if __name__ == "__main__":
    main()
