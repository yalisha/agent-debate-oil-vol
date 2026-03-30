"""
Further optimization experiments:
1. Regime-weighted loss (upweight extreme regimes during training)
2. Post-hoc GAT+MLP stacking (regime-conditional blend)
3. Rolling accuracy features (lagged prediction error as input)
4. DropEdge regularization during training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20

from optimized_gat import (
    OptimizedGATModel, MultiHeadGATLayer, dm_test_hac
)
from final_comparison import OptimizedMLPModel, train_mlp


# ══════════════════════════════════════════════════════════
# 1. REGIME-WEIGHTED LOSS
# ══════════════════════════════════════════════════════════

def train_gat_regime_weighted(model, node_feats, context, targets, base_preds,
                              regime_feats, train_idx, persist_vol,
                              n_epochs=250, lr=0.003):
    """Train with sample weights: upweight extreme regimes."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    # Sample weights: upweight low/high vol regimes
    pv = persist_vol[train_idx]
    weights = np.ones(len(train_idx))
    weights[pv < 0.20] = 2.0   # low regime
    weights[pv > 0.55] = 2.0   # high regime
    w = torch.FloatTensor(weights)
    w = w / w.mean()  # normalize so mean weight = 1

    best_loss, best_state = float('inf'), None

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, _, _, _, _ = model(nf, ctx, bp, rf)
        loss = (w * (pred - y) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_loss


# ══════════════════════════════════════════════════════════
# 2. DROP-EDGE REGULARIZATION
# ══════════════════════════════════════════════════════════

class DropEdgeGATModel(OptimizedGATModel):
    """GAT with random edge dropout during training for regularization."""

    def __init__(self, *args, drop_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_rate = drop_rate

    def forward(self, node_feats, context, base_preds, regime_feats):
        edge_mask = self.get_effective_graph()

        # DropEdge during training
        if self.training and self.drop_rate > 0:
            drop_mask = (torch.rand_like(edge_mask) > self.drop_rate).float()
            # Keep self-loops
            drop_mask = drop_mask * (1 - torch.eye(N_AGENTS, device=edge_mask.device)) + \
                        torch.eye(N_AGENTS, device=edge_mask.device)
            edge_mask = edge_mask * drop_mask

        skip = self.skip_proj(node_feats)
        h, _ = self.gat1(node_feats, edge_mask)
        h = self.norm1(h + skip)
        h = F.relu(h)

        h2, attn = self.gat2(h, edge_mask)
        h = self.norm2(h2 + h)
        h = F.relu(h)

        graph_embed = h.reshape(h.size(0), -1)
        combined = torch.cat([graph_embed, context], dim=-1)

        w_base = torch.softmax(self.base_head(combined), dim=-1)
        w_regime = torch.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate, edge_mask, attn


def train_gat(model, node_feats, context, targets, base_preds, regime_feats,
              train_idx, n_epochs=250, lr=0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    best_loss, best_state = float('inf'), None
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred, _, _, _, _ = model(nf, ctx, bp, rf)
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
    return best_loss


# ══════════════════════════════════════════════════════════
# 3. ENHANCED FEATURES (rolling accuracy)
# ══════════════════════════════════════════════════════════

def add_rolling_accuracy_features(df_clean, context, context_dim):
    """Add lagged rolling accuracy of debate/single predictions.

    Uses 20-day lag to avoid look-ahead (since target is 20-day forward RV).
    """
    actual = df_clean['actual_vol'].values
    debate = df_clean['debate_vol'].values
    single = df_clean['single_vol'].values

    # Squared errors (shifted by 20 to avoid look-ahead)
    debate_se = np.full(len(df_clean), np.nan)
    single_se = np.full(len(df_clean), np.nan)

    for t in range(20, len(df_clean)):
        debate_se[t] = (debate[t-20] - actual[t-20]) ** 2
        single_se[t] = (single[t-20] - actual[t-20]) ** 2

    # Rolling mean of past errors (window=20, properly lagged)
    debate_err_ma = pd.Series(debate_se).rolling(20, min_periods=5).mean().values
    single_err_ma = pd.Series(single_se).rolling(20, min_periods=5).mean().values

    # Relative accuracy: negative = debate better, positive = single better
    rel_accuracy = np.nan_to_num(debate_err_ma - single_err_ma, nan=0.0)

    # Add to context
    new_features = np.column_stack([
        np.nan_to_num(debate_err_ma, nan=0.02),
        np.nan_to_num(single_err_ma, nan=0.02),
        rel_accuracy,
    ])

    enhanced_context = np.column_stack([context, new_features])
    return enhanced_context, context_dim + 3


# ══════════════════════════════════════════════════════════
# WALK-FORWARD RUNNERS
# ══════════════════════════════════════════════════════════

def run_gat_seed(seed, df_clean, node_feats, context, targets, base_preds,
                 regime_feats, node_feat_dim, context_dim, n_base,
                 model_cls=OptimizedGATModel, train_fn=train_gat,
                 pv_for_weighting=None, **model_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = model_cls(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=16, n_heads=4, top_k=3, prior_adj=None,
            **model_kwargs
        )

        if train_fn == train_gat_regime_weighted:
            train_fn(model, node_feats, context, targets, base_preds,
                     regime_feats, train_idx, pv_for_weighting)
        else:
            train_fn(model, node_feats, context, targets, base_preds,
                     regime_feats, train_idx)

        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            pred, _, _, _, _ = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()

    return preds


def run_mlp_seed(seed, df_clean, node_feats, context, targets, base_preds,
                 regime_feats, node_feat_dim, context_dim, n_base):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = OptimizedMLPModel(node_feat_dim, context_dim, n_base=n_base)
        train_mlp(model, node_feats, context, targets, base_preds,
                  regime_feats, train_idx)

        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            pred, _, _ = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()

    return preds


def evaluate_method(name, pred_fn, df_clean, node_feats, context, targets,
                    base_preds, regime_feats, har, persist_vol,
                    node_feat_dim, context_dim, n_base, seeds, **kwargs):
    """Evaluate a method with multi-seed ensemble."""
    all_preds = []
    seed_rmses = []

    for seed in seeds:
        preds = pred_fn(seed, df_clean, node_feats, context, targets,
                        base_preds, regime_feats, node_feat_dim, context_dim,
                        n_base, **kwargs)
        valid = ~np.isnan(preds)
        rmse = np.sqrt(np.mean((preds[valid] - targets[valid])**2))
        seed_rmses.append(rmse)
        all_preds.append(preds)

    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    y = targets[valid]
    rmse_ens = np.sqrt(np.mean((ens[valid] - y)**2))
    dm, p = dm_test_hac(ens[valid] - y, har[valid] - y)

    # Regime breakdown
    regimes = pd.cut(persist_vol[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    regime_rmse = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            regime_rmse[r] = np.sqrt(np.mean((ens[valid][mask] - y[mask])**2))

    print(f"  {name}:")
    print(f"    Per-seed: {np.mean(seed_rmses):.4f} +/- {np.std(seed_rmses):.4f}")
    print(f"    Ensemble: {rmse_ens:.4f}, DM_HAC={dm:.3f} (p={p:.4f})")
    print(f"    Regime:   low={regime_rmse.get('low', 0):.4f} "
          f"normal={regime_rmse.get('normal', 0):.4f} "
          f"elev={regime_rmse.get('elevated', 0):.4f} "
          f"high={regime_rmse.get('high', 0):.4f}")

    return {
        'name': name, 'ensemble_rmse': rmse_ens, 'dm_hac': dm, 'p_hac': p,
        'perseed_mean': np.mean(seed_rmses), 'perseed_std': np.std(seed_rmses),
        'ensemble_preds': ens, **{f'rmse_{k}': v for k, v in regime_rmse.items()}
    }


# ══════════════════════════════════════════════════════════
# 4. POST-HOC STACKING
# ══════════════════════════════════════════════════════════

def regime_conditional_stack(gat_preds, mlp_preds, persist_vol, targets, har):
    """Walk-forward regime-conditional stacking of GAT and MLP."""
    n = len(targets)
    stacked = np.full(n, np.nan)

    valid_mask = ~np.isnan(gat_preds) & ~np.isnan(mlp_preds)

    for rp in range(315, n, 63):  # start later (need GAT/MLP predictions first)
        chunk_end = min(rp + 63, n)
        test_idx = np.arange(rp, chunk_end)

        # Learn weights on past predictions
        train_mask = valid_mask.copy()
        train_mask[embargoed_train_end(rp):] = False
        train_idx = np.where(train_mask)[0]

        if len(train_idx) < 30:
            stacked[test_idx] = gat_preds[test_idx]
            continue

        # Regime-conditional weights: learn alpha per regime
        pv_train = persist_vol[train_idx]
        y_train = targets[train_idx]
        g_train = gat_preds[train_idx]
        m_train = mlp_preds[train_idx]

        # For each regime, find optimal blend weight
        alphas = {}
        for regime, (lo, hi) in {'low': (0, 0.20), 'normal': (0.20, 0.35),
                                  'elevated': (0.35, 0.55), 'high': (0.55, 100)}.items():
            mask = (pv_train >= lo) & (pv_train < hi)
            if mask.sum() < 10:
                alphas[regime] = 0.8  # default: mostly GAT
            else:
                # Grid search for optimal alpha
                best_a, best_rmse = 1.0, float('inf')
                for a in np.arange(0, 1.05, 0.05):
                    blend = a * g_train[mask] + (1-a) * m_train[mask]
                    rmse = np.sqrt(np.mean((blend - y_train[mask])**2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_a = a
                alphas[regime] = best_a

        # Apply to test period
        for t in test_idx:
            pv = persist_vol[t]
            if pv < 0.20:
                a = alphas['low']
            elif pv < 0.35:
                a = alphas['normal']
            elif pv < 0.55:
                a = alphas['elevated']
            else:
                a = alphas['high']
            stacked[t] = a * gat_preds[t] + (1-a) * mlp_preds[t]

    valid = ~np.isnan(stacked)
    y = targets[valid]
    rmse = np.sqrt(np.mean((stacked[valid] - y)**2))
    dm, p = dm_test_hac(stacked[valid] - y, har[valid] - y)

    regimes = pd.cut(persist_vol[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    print(f"\n  Regime-Conditional Stack:")
    print(f"    RMSE: {rmse:.4f}, DM_HAC={dm:.3f} (p={p:.4f})")
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            r_rmse = np.sqrt(np.mean((stacked[valid][mask] - y[mask])**2))
            print(f"    {r}: {r_rmse:.4f} (n={mask.sum()})")

    return stacked, rmse, dm, p


def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("FURTHER OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"Label embargo: {LABEL_EMBARGO} trading days for forward-looking target")

    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
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

    har = df_clean['har_vol'].values
    seeds = [42, 123, 456, 789, 1024]

    results = []

    # ── 1. Baseline: Learned GAT (reference) ──
    print(f"\n{'='*70}")
    print("EXP 1: LEARNED GAT (reference)")
    print(f"{'='*70}")
    r1 = evaluate_method(
        "Learned GAT", run_gat_seed,
        df_clean, node_feats, context, targets, base_preds,
        regime_feats, har, persist_vol, node_feat_dim, context_dim,
        n_base, seeds
    )
    results.append(r1)

    # ── 2. Regime-weighted loss ──
    print(f"\n{'='*70}")
    print("EXP 2: REGIME-WEIGHTED LOSS (2x weight on extreme regimes)")
    print(f"{'='*70}")
    r2 = evaluate_method(
        "Regime-Weighted GAT", run_gat_seed,
        df_clean, node_feats, context, targets, base_preds,
        regime_feats, har, persist_vol, node_feat_dim, context_dim,
        n_base, seeds,
        train_fn=train_gat_regime_weighted, pv_for_weighting=persist_vol
    )
    results.append(r2)

    # ── 3. DropEdge GAT ──
    print(f"\n{'='*70}")
    print("EXP 3: DROP-EDGE GAT (20% edge dropout during training)")
    print(f"{'='*70}")
    r3 = evaluate_method(
        "DropEdge GAT", run_gat_seed,
        df_clean, node_feats, context, targets, base_preds,
        regime_feats, har, persist_vol, node_feat_dim, context_dim,
        n_base, seeds,
        model_cls=DropEdgeGATModel, drop_rate=0.2
    )
    results.append(r3)

    # ── 4. Enhanced features ──
    print(f"\n{'='*70}")
    print("EXP 4: ENHANCED FEATURES (rolling prediction accuracy)")
    print(f"{'='*70}")
    enh_context, enh_context_dim = add_rolling_accuracy_features(df_clean, context, context_dim)
    r4 = evaluate_method(
        "Enhanced Features GAT", run_gat_seed,
        df_clean, node_feats, enh_context, targets, base_preds,
        regime_feats, har, persist_vol, node_feat_dim, enh_context_dim,
        n_base, seeds
    )
    results.append(r4)

    # ── 5. MLP baseline (for stacking) ──
    print(f"\n{'='*70}")
    print("EXP 5: MLP BASELINE (for stacking)")
    print(f"{'='*70}")
    r5 = evaluate_method(
        "MLP no-graph", run_mlp_seed,
        df_clean, node_feats, context, targets, base_preds,
        regime_feats, har, persist_vol, node_feat_dim, context_dim,
        n_base, seeds
    )
    results.append(r5)

    # ── 6. Post-hoc stacking ──
    print(f"\n{'='*70}")
    print("EXP 6: REGIME-CONDITIONAL STACKING (GAT + MLP)")
    print(f"{'='*70}")
    stacked, stack_rmse, stack_dm, stack_p = regime_conditional_stack(
        r1['ensemble_preds'], r5['ensemble_preds'],
        persist_vol, targets, har
    )

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Method':<30s} {'RMSE':>7s} {'DM_HAC':>8s} {'p':>7s} "
          f"{'low':>7s} {'norm':>7s} {'elev':>7s} {'high':>7s}")
    print(f"  {'─'*85}")

    for r in sorted(results, key=lambda x: x['ensemble_rmse']):
        print(f"  {r['name']:<30s} {r['ensemble_rmse']:>7.4f} {r['dm_hac']:>8.3f} "
              f"{r['p_hac']:>7.4f} "
              f"{r.get('rmse_low', 0):>7.4f} {r.get('rmse_normal', 0):>7.4f} "
              f"{r.get('rmse_elevated', 0):>7.4f} {r.get('rmse_high', 0):>7.4f}")

    print(f"  {'Stacked (GAT+MLP)':<30s} {stack_rmse:>7.4f} {stack_dm:>8.3f} {stack_p:>7.4f}")

    # Save
    df_out = pd.DataFrame([{k: v for k, v in r.items() if k != 'ensemble_preds'} for r in results])
    df_out.to_csv(BASE / "results" / "further_optimization_results.csv", index=False)
    print(f"\nSaved to results/further_optimization_results.csv")


if __name__ == "__main__":
    main()
