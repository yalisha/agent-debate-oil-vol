"""
Final model: Optimized GAT with DropEdge + Enhanced Features.

Combines the best findings from all optimization rounds:
- Multi-head attention (4 heads) with learned sparse graph
- DropEdge regularization (reduces seed variance)
- Enhanced features (rolling prediction accuracy, helps high-vol regime)
- Regime-gated dual head
- Skip connection + LayerNorm

5-seed ensemble with HAC DM test.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20

from optimized_gat import MultiHeadGATLayer, dm_test_hac
from further_optimization import add_rolling_accuracy_features


# ══════════════════════════════════════════════════════════
# FINAL MODEL
# ══════════════════════════════════════════════════════════

class FinalGATModel(nn.Module):
    """Production model: DropEdge + Enhanced Features + all best practices."""

    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16,
                 n_heads=4, top_k=3, drop_rate=0.2):
        super().__init__()
        self.n_base = n_base
        self.drop_rate = drop_rate

        # Learned graph
        self.edge_logits = nn.Parameter(torch.randn(N_AGENTS, N_AGENTS) * 0.1)

        # Multi-head GAT
        self.gat1 = MultiHeadGATLayer(node_feat_dim, hidden_dim, n_heads=n_heads, top_k=top_k)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, top_k=top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Skip connection
        self.skip_proj = nn.Linear(node_feat_dim, hidden_dim)

        graph_dim = N_AGENTS * hidden_dim + context_dim

        # Regime-gated dual head
        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def get_effective_graph(self):
        gates = torch.sigmoid(self.edge_logits)
        gates = gates * (1 - torch.eye(N_AGENTS, device=gates.device)) + \
                torch.eye(N_AGENTS, device=gates.device)
        return gates

    def get_sparsity_stats(self):
        with torch.no_grad():
            gates = torch.sigmoid(self.edge_logits)
            mask = 1.0 - torch.eye(N_AGENTS)
            return ((gates * mask) > 0.5).sum().item()

    def forward(self, node_feats, context, base_preds, regime_feats):
        edge_mask = self.get_effective_graph()

        # DropEdge during training
        if self.training and self.drop_rate > 0:
            drop = (torch.rand_like(edge_mask) > self.drop_rate).float()
            drop = drop * (1 - torch.eye(N_AGENTS, device=edge_mask.device)) + \
                   torch.eye(N_AGENTS, device=edge_mask.device)
            edge_mask = edge_mask * drop

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


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train_model(model, node_feats, context, targets, base_preds, regime_feats,
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
# EVALUATION
# ══════════════════════════════════════════════════════════

def run_seed(seed, df_clean, node_feats, context, targets, base_preds,
             regime_feats, node_feat_dim, context_dim, n_base):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)
    all_gates = np.full(n, np.nan)
    all_sparsity = []

    for rp in range(252, n, 63):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = FinalGATModel(node_feat_dim, context_dim, n_base=n_base)
        train_model(model, node_feats, context, targets, base_preds,
                    regime_feats, train_idx)

        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            pred, weights, gate, edge_mask, attn = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()
        all_gates[test_idx] = gate.squeeze(-1).numpy()
        all_sparsity.append(model.get_sparsity_stats())

    return preds, all_gates, all_sparsity


def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("FINAL MODEL: DropEdge GAT + Enhanced Features")
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

    # Enhanced features
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

    har = df_clean['har_vol'].values
    debate = df_clean['debate_vol'].values
    single = df_clean['single_vol'].values

    seeds = [42, 123, 456, 789, 1024]

    # ── MAIN EVALUATION ──
    print(f"\n{'='*70}")
    print("5-SEED EVALUATION")
    print(f"{'='*70}")

    all_preds = []
    seed_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}...", end=" ", flush=True)
        preds, gates, sparsity = run_seed(
            seed, df_clean, node_feats, enh_context, targets, base_preds,
            regime_feats, node_feat_dim, enh_context_dim, n_base
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        mae = np.mean(np.abs(p - y))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        mean_gate = np.nanmean(gates[valid])
        mean_sp = np.mean(sparsity)

        print(f"RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={pval:.4f}), "
              f"edges={mean_sp:.0f}/42, gate={mean_gate:.3f}")
        seed_results.append({
            'seed': seed, 'rmse': rmse, 'mae': mae,
            'dm_hac': dm, 'p_hac': pval,
            'gate': mean_gate, 'edges': mean_sp
        })
        all_preds.append(preds)

    # Ensemble
    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    y_v = targets[valid]
    rmse_ens = np.sqrt(np.mean((ens[valid] - y_v)**2))
    mae_ens = np.mean(np.abs(ens[valid] - y_v))
    dm_ens, p_ens = dm_test_hac(ens[valid] - y_v, har[valid] - y_v)

    rmses = [r['rmse'] for r in seed_results]
    print(f"\n  {'='*50}")
    print(f"  Per-seed: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"  Ensemble: RMSE={rmse_ens:.4f} MAE={mae_ens:.4f}")
    print(f"  DM_HAC vs HAR: {dm_ens:.3f} (p={p_ens:.4f})")

    # ── REGIME BREAKDOWN ──
    print(f"\n{'='*70}")
    print("REGIME BREAKDOWN (ensemble)")
    print(f"{'='*70}")

    regimes = pd.cut(persist_vol[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])

    print(f"\n  {'Regime':<12s} {'Final':>8s} {'GAT-v3':>8s} {'HAR':>8s} {'Debate':>8s} {'n':>6s}")
    print(f"  {'─'*55}")
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            r_fin = np.sqrt(np.mean((ens[valid][mask] - y_v[mask])**2))
            r_har = np.sqrt(np.mean((har[valid][mask] - y_v[mask])**2))
            r_deb = np.sqrt(np.mean((debate[valid][mask] - y_v[mask])**2))
            print(f"  {r:<12s} {r_fin:>8.4f} {'':>8s} {r_har:>8.4f} {r_deb:>8.4f} {mask.sum():>6d}")

    # ── FULL COMPARISON ──
    print(f"\n{'='*70}")
    print("FULL COMPARISON (all HAC-corrected)")
    print(f"{'='*70}")

    print(f"\n  {'Model':<35s} {'RMSE':>7s} {'MAE':>7s} {'DM_HAC':>9s} {'p':>8s} {'Sig':>4s}")
    print(f"  {'─'*70}")

    comparisons = [
        ('Final (DropEdge+EnhFeats)', rmse_ens, mae_ens, dm_ens, p_ens),
    ]
    for name, col in [('Debate (naive)', 'debate_vol'), ('Single agent', 'single_vol'),
                      ('HAR', 'har_vol'), ('Persistence', 'persist_vol')]:
        vals = df_clean[col].values[valid]
        rmse = np.sqrt(np.mean((vals - y_v)**2))
        mae = np.mean(np.abs(vals - y_v))
        if name != 'HAR':
            dm, p = dm_test_hac(vals - y_v, har[valid] - y_v)
        else:
            dm, p = 0, 1
        comparisons.append((name, rmse, mae, dm, p))

    for name, rmse, mae, dm, p in sorted(comparisons, key=lambda x: x[1]):
        sig = '*' if p < 0.05 else ''
        print(f"  {name:<35s} {rmse:>7.4f} {mae:>7.4f} {dm:>9.3f} {p:>8.4f} {sig:>4s}")

    print(f"\n  Reference from optimization history:")
    print(f"    Learned GAT (v3, no enh feats):  0.1226")
    print(f"    DropEdge GAT (no enh feats):     0.1228")
    print(f"    Enhanced Features GAT (no DE):    0.1228")
    print(f"    Post-hoc stacking:               0.1201")

    # ── DM vs Learned GAT (for significance of improvement) ──
    # Can't compute directly without Learned GAT predictions, but report difference
    print(f"\n  Improvement vs Learned GAT v3:     {0.1226 - rmse_ens:.4f} ({(0.1226 - rmse_ens)/0.1226*100:.1f}%)")

    # Save
    df_out = pd.DataFrame(seed_results)
    df_out.to_csv(BASE / "results" / "final_model_results.csv", index=False)
    print(f"\nSaved to results/final_model_results.csv")


if __name__ == "__main__":
    main()
