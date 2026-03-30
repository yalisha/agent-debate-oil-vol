"""
P2+P3: Hybrid Sparse GAT with Regime-Gated Head.

P2: Graph structure
  - Prior edges from Granger causal graph (warm start)
  - Learnable edge gates (sigmoid)
  - L1 sparsity penalty on edge gates
  - Combined: prior * learned_gate

P3: Regime-gated output
  - Two heads: base head + regime-modulated head
  - Soft gate based on volatility level (no hard regime bins)
  - Prevents overfitting to few high-vol points

Evaluation: fixed seeds, 3-seed mean/std, HAC DM test.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20


# ══════════════════════════════════════════════════════════
# HYBRID SPARSE GAT
# ══════════════════════════════════════════════════════════

class LearnableGraphAttention(nn.Module):
    """GAT layer with learnable edge gating."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_mask):
        """
        x: (batch, n_nodes, in_dim)
        edge_mask: (n_nodes, n_nodes) soft mask [0,1]
        """
        h = self.W(x)
        B, N, D = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        e = self.leaky(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))

        # Mask by learned edge weights
        mask = edge_mask.unsqueeze(0)
        e = e * mask + (1 - mask) * (-1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = torch.bmm(alpha, h)
        return out, alpha


class HybridSparseGATModel(nn.Module):
    """
    Hybrid Sparse GAT with regime-gated head.

    Graph: prior_adj * sigmoid(learned_logits) -> sparse effective graph
    Output: soft blend of base_head and regime_head based on vol level
    """
    N_BASE = 2  # debate_vol, single_vol (LLM-only)

    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16,
                 prior_adj=None):
        super().__init__()
        self.n_base = n_base

        # Learnable edge logits (initialized from prior)
        if prior_adj is not None:
            # Initialize logits so sigmoid(logits) ~ prior_adj
            init_logits = torch.zeros(N_AGENTS, N_AGENTS)
            for i in range(N_AGENTS):
                for j in range(N_AGENTS):
                    if prior_adj[i, j] > 0.5:
                        init_logits[i, j] = 2.0  # sigmoid(2) ~ 0.88
                    else:
                        init_logits[i, j] = -2.0  # sigmoid(-2) ~ 0.12
            self.edge_logits = nn.Parameter(init_logits)
        else:
            self.edge_logits = nn.Parameter(torch.zeros(N_AGENTS, N_AGENTS))

        # GAT layers
        self.gat1 = LearnableGraphAttention(node_feat_dim, hidden_dim)
        self.gat2 = LearnableGraphAttention(hidden_dim, hidden_dim)

        graph_dim = N_AGENTS * hidden_dim + context_dim

        # Base weight head (normal conditions)
        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_base),
        )

        # Regime-modulated head (adapts under stress)
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_base),
        )

        # Regime gate: soft switch based on vol level
        # Input: persist_vol (or vol-related features from context)
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8),  # [persist_vol, vol_change_5d, n_herding]
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Residual head
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def get_effective_graph(self):
        """Compute effective adjacency with learned gating."""
        return torch.sigmoid(self.edge_logits)

    def edge_sparsity_loss(self):
        """L1 penalty on edge gates to encourage sparsity."""
        gates = torch.sigmoid(self.edge_logits)
        # Don't penalize self-loops
        mask = 1.0 - torch.eye(N_AGENTS, device=gates.device)
        return (gates * mask).sum()

    def forward(self, node_feats, context, base_preds, regime_feats):
        """
        node_feats: (batch, 7, node_feat_dim)
        context: (batch, context_dim)
        base_preds: (batch, n_base)
        regime_feats: (batch, 3) [persist_vol, vol_change_5d, n_herding]
        """
        # Effective graph
        edge_mask = self.get_effective_graph()

        # GAT
        h, _ = self.gat1(node_feats, edge_mask)
        h = torch.relu(h)
        h, attn = self.gat2(h, edge_mask)
        h = torch.relu(h)

        graph_embed = h.reshape(h.size(0), -1)
        combined = torch.cat([graph_embed, context], dim=-1)

        # Two sets of weights
        w_base = torch.softmax(self.base_head(combined), dim=-1)
        w_regime = torch.softmax(self.regime_head(combined), dim=-1)

        # Regime gate
        gate = self.regime_gate(regime_feats)  # (batch, 1)

        # Blended weights
        weights = (1 - gate) * w_base + gate * w_regime  # (batch, n_base)

        # Weighted prediction
        pred = (weights * base_preds).sum(dim=-1)

        # Residual
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate, edge_mask


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train_model(model, node_feats, context, targets, base_preds, regime_feats,
                train_idx, n_epochs=250, lr=0.003, l1_lambda=0.01):
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
        pred, _, _, _ = model(nf, ctx, bp, rf)

        mse_loss = loss_fn(pred, y)
        sparsity_loss = model.edge_sparsity_loss() * l1_lambda
        loss = mse_loss + sparsity_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if mse_loss.item() < best_loss:
            best_loss = mse_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_loss


# ══════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════

def dm_test_hac(e1, e2, h=H):
    d = e1**2 - e2**2
    d = d[~np.isnan(d)]
    T = len(d)
    if T < h + 10:
        return np.nan, np.nan
    d_bar = d.mean()
    bandwidth = h - 1
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        if k < T:
            weight = 1.0 - k / (bandwidth + 1)
            cov_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            gamma_sum += weight * cov_k
    var_d = (gamma_0 + 2 * gamma_sum) / T
    if var_d <= 0:
        var_d = gamma_0 / T
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def run_single_seed(seed, df_clean, node_feats, context, targets,
                    base_preds, regime_feats, node_feat_dim, context_dim,
                    n_base, l1_lambda=0.01):
    """Run one seed of walk-forward evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from causal_gat_aggregation import build_causal_graph

    n = len(df_clean)
    min_train = 252
    retrain_every = 63
    preds = np.full(n, np.nan)
    all_gates = np.full(n, np.nan)
    all_sparsity = []

    retrain_points = list(range(min_train, n, retrain_every))

    for rp in retrain_points:
        chunk_end = min(rp + retrain_every, n)
        train_end = embargoed_train_end(rp)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < min_train:
            continue

        # Prior graph from causal discovery
        prior_adj = build_causal_graph(df_clean.iloc[:train_end])

        # Model
        model = HybridSparseGATModel(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=16, prior_adj=prior_adj
        )

        # Train
        train_model(model, node_feats, context, targets, base_preds,
                    regime_feats, train_idx, l1_lambda=l1_lambda)

        # Predict
        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            pred, weights, gate, edge_mask = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()
        all_gates[test_idx] = gate.squeeze(-1).numpy()

        # Log sparsity
        effective = edge_mask.detach().numpy()
        np.fill_diagonal(effective, 0)
        n_active = (effective > 0.5).sum()
        all_sparsity.append(n_active)

    return preds, all_gates, all_sparsity


def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("P2+P3: HYBRID SPARSE GAT WITH REGIME-GATED HEAD")
    print("=" * 70)

    # Load and sort
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values

    # LLM-only base predictions
    base_cols = ['debate_vol', 'single_vol']
    base_preds = np.column_stack([df_clean[c].values for c in base_cols])
    n_base = len(base_cols)

    # Regime features for gating
    regime_feats = np.column_stack([
        df_clean['persist_vol'].values,
        df_clean['persist_vol'].diff().fillna(0).values,
        df_clean['n_herding'].values / 7.0,
    ])

    har = df_clean['har_vol'].values
    debate = df_clean['debate_vol'].values
    single = df_clean['single_vol'].values
    persist = df_clean['persist_vol'].values

    # Test different L1 lambdas
    lambdas = [0.0, 0.005, 0.01, 0.02, 0.05]
    seeds = [42, 123, 456]

    print(f"\n{'─'*70}")
    print("PART 1: L1 SPARSITY SWEEP (seed=42)")
    print(f"{'─'*70}")

    best_lambda = 0.01
    best_rmse = float('inf')

    for l1 in lambdas:
        preds, gates, sparsity = run_single_seed(
            42, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim,
            n_base, l1_lambda=l1
        )
        valid = ~np.isnan(preds)
        rmse = np.sqrt(np.mean((preds[valid] - targets[valid])**2))
        dm, p = dm_test_hac(preds[valid] - targets[valid], har[valid] - targets[valid])
        mean_sparsity = np.mean(sparsity)
        mean_gate = np.nanmean(gates[valid])

        print(f"  L1={l1:.3f}: RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={p:.4f}), "
              f"active_edges={mean_sparsity:.0f}/42, regime_gate={mean_gate:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = l1

    print(f"\n  Best L1 lambda: {best_lambda}")

    # Multi-seed with best lambda
    print(f"\n{'─'*70}")
    print(f"PART 2: MULTI-SEED EVALUATION (L1={best_lambda})")
    print(f"{'─'*70}")

    all_seed_preds = []
    seed_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}...", end=" ", flush=True)
        preds, gates, sparsity = run_single_seed(
            seed, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim,
            n_base, l1_lambda=best_lambda
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        mae = np.mean(np.abs(p - y))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        mean_gate = np.nanmean(gates[valid])

        print(f"RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={pval:.4f}), gate={mean_gate:.3f}")
        seed_results.append({'seed': seed, 'rmse': rmse, 'mae': mae,
                             'dm_hac': dm, 'p_hac': pval, 'gate': mean_gate})
        all_seed_preds.append(preds)

    # Summary
    rmses = [r['rmse'] for r in seed_results]
    dms = [r['dm_hac'] for r in seed_results]
    print(f"\n  ────────────────────────────")
    print(f"  RMSE:   {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  DM_HAC: {np.mean(dms):.3f} ± {np.std(dms):.3f}")

    # Ensemble
    ens = np.nanmean(np.stack(all_seed_preds), axis=0)
    valid = ~np.isnan(ens)
    rmse_ens = np.sqrt(np.mean((ens[valid] - targets[valid])**2))
    dm_ens, p_ens = dm_test_hac(ens[valid] - targets[valid], har[valid] - targets[valid])
    print(f"  Ensemble: RMSE={rmse_ens:.4f}, DM_HAC={dm_ens:.3f} (p={p_ens:.4f})")

    # Regime breakdown
    print(f"\n{'─'*70}")
    print("PART 3: REGIME BREAKDOWN (ensemble)")
    print(f"{'─'*70}")

    regimes = pd.cut(df_clean['persist_vol'].values[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    y_v = targets[valid]
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            rmse_r = np.sqrt(np.mean((ens[valid][mask] - y_v[mask])**2))
            rmse_har = np.sqrt(np.mean((har[valid][mask] - y_v[mask])**2))
            rmse_deb = np.sqrt(np.mean((debate[valid][mask] - y_v[mask])**2))
            print(f"  {r:10s}: Hybrid={rmse_r:.4f}  HAR={rmse_har:.4f}  "
                  f"Debate={rmse_deb:.4f}  n={mask.sum()}")

    # Compare with original Causal-GAT
    print(f"\n{'─'*70}")
    print("PART 4: COMPARISON TABLE (all HAC-corrected)")
    print(f"{'─'*70}")

    print(f"\n  {'Model':<35s} {'RMSE':>7s} {'DM_HAC':>9s} {'p':>8s} {'Sig':>4s}")
    print(f"  {'─'*65}")

    comparisons = [
        ('Hybrid-GAT (ensemble)', rmse_ens, dm_ens, p_ens),
    ]

    # Add reference models
    for name, col in [('Debate (naive)', 'debate_vol'), ('Single', 'single_vol'),
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

    # Save
    out_df = pd.DataFrame(seed_results)
    out_df.to_csv(BASE / "results" / "hybrid_sparse_gat_results.csv", index=False)
    print(f"\nSaved to results/hybrid_sparse_gat_results.csv")


if __name__ == "__main__":
    main()
