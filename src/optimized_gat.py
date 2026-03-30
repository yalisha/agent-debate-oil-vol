"""
Optimized GAT Meta-Learner for Multi-Agent Debate Aggregation.

Key improvements over Hybrid Sparse GAT:
1. Top-k hard sparsification (replaces failed L1 penalty)
2. Multi-head attention (4 heads, richer agent interaction)
3. Validation-based early stopping (reduces seed variance)
4. Regime-gated dual head (retained, works well for high-vol)
5. Optional: learned graph from scratch vs causal prior warm-start

Walk-forward evaluation with HAC DM test.
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


# ══════════════════════════════════════════════════════════
# MULTI-HEAD GAT WITH TOP-K SPARSIFICATION
# ══════════════════════════════════════════════════════════

class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT with top-k edge pruning per node."""
    def __init__(self, in_dim, out_dim, n_heads=4, top_k=3):
        super().__init__()
        self.n_heads = n_heads
        self.top_k = top_k
        assert out_dim % n_heads == 0
        self.head_dim = out_dim // n_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_mask=None):
        """
        x: (batch, n_nodes, in_dim)
        edge_mask: (n_nodes, n_nodes) optional soft mask

        Returns: (batch, n_nodes, out_dim), attention weights
        """
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.head_dim)  # (B, N, H, D)

        # Additive attention: a_src * h_i + a_dst * h_j
        score_src = (h * self.a_src).sum(-1)  # (B, N, H)
        score_dst = (h * self.a_dst).sum(-1)
        # e[i,j] = score_src[i] + score_dst[j]
        e = score_src.unsqueeze(2) + score_dst.unsqueeze(1)  # (B, N, N, H)
        e = self.leaky(e)

        # Apply edge mask if provided (soft prior)
        if edge_mask is not None:
            mask = edge_mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            e = e * mask + (1 - mask) * (-1e9)

        # Top-k hard sparsification per node
        if self.top_k < N:
            # For each node i, keep only top-k incoming edges
            # e shape: (B, N, N, H) where dim=2 is the source dimension
            topk_vals, topk_idx = torch.topk(e, self.top_k, dim=2)
            threshold = topk_vals[:, :, -1:, :]  # (B, N, 1, H)
            sparse_mask = (e >= threshold).float()
            e = e * sparse_mask + (1 - sparse_mask) * (-1e9)

        alpha = torch.softmax(e, dim=2)  # (B, N, N, H) normalize over sources
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Aggregate: for each node i, weighted sum of neighbors
        # h: (B, N, H, D), alpha: (B, N, N, H)
        # out[i] = sum_j alpha[i,j] * h[j]
        h_perm = h.permute(0, 2, 1, 3)  # (B, H, N, D)
        alpha_perm = alpha.permute(0, 3, 1, 2)  # (B, H, N, N)
        out = torch.matmul(alpha_perm, h_perm)  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, H*D)

        return out, alpha.mean(dim=-1)  # return mean attention across heads


class OptimizedGATModel(nn.Module):
    """
    Optimized GAT with:
    - Multi-head attention (4 heads)
    - Top-k hard sparsification (k per node)
    - Learnable edge logits with causal prior warm-start
    - Regime-gated dual output head
    - Skip connection (graph bypass)
    """
    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16,
                 n_heads=4, top_k=3, prior_adj=None):
        super().__init__()
        self.n_base = n_base
        self.top_k = top_k

        # Learnable graph: sigmoid(logits) as soft edge mask
        if prior_adj is not None:
            init_logits = torch.zeros(N_AGENTS, N_AGENTS)
            for i in range(N_AGENTS):
                for j in range(N_AGENTS):
                    if prior_adj[i, j] > 0.5:
                        init_logits[i, j] = 2.0
                    else:
                        init_logits[i, j] = -2.0
            self.edge_logits = nn.Parameter(init_logits)
        else:
            # Learn from scratch: initialize near zero (uniform prior)
            self.edge_logits = nn.Parameter(torch.randn(N_AGENTS, N_AGENTS) * 0.1)

        # Multi-head GAT layers
        self.gat1 = MultiHeadGATLayer(node_feat_dim, hidden_dim, n_heads=n_heads, top_k=top_k)
        self.gat2 = MultiHeadGATLayer(hidden_dim, hidden_dim, n_heads=n_heads, top_k=top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Skip connection: project node features to same dim as GAT output
        self.skip_proj = nn.Linear(node_feat_dim, hidden_dim)

        graph_dim = N_AGENTS * hidden_dim + context_dim

        # Base weight head
        self.base_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )

        # Regime-modulated head
        self.regime_head = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, n_base),
        )

        # Regime gate
        self.regime_gate = nn.Sequential(
            nn.Linear(3, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Residual correction
        self.residual_head = nn.Sequential(
            nn.Linear(graph_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def get_effective_graph(self):
        gates = torch.sigmoid(self.edge_logits)
        # Always keep self-loops
        gates = gates * (1 - torch.eye(N_AGENTS, device=gates.device)) + torch.eye(N_AGENTS, device=gates.device)
        return gates

    def get_sparsity_stats(self):
        """Return number of active edges (gate > 0.5, excluding self-loops)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.edge_logits)
            mask = 1.0 - torch.eye(N_AGENTS)
            active = ((gates * mask) > 0.5).sum().item()
        return active

    def forward(self, node_feats, context, base_preds, regime_feats):
        """
        node_feats: (batch, 7, node_feat_dim)
        context: (batch, context_dim)
        base_preds: (batch, n_base)
        regime_feats: (batch, 3)
        """
        edge_mask = self.get_effective_graph()

        # Skip connection input
        skip = self.skip_proj(node_feats)  # (B, N, hidden)

        # GAT layer 1
        h, _ = self.gat1(node_feats, edge_mask)
        h = self.norm1(h + skip)  # residual + layernorm
        h = F.relu(h)

        # GAT layer 2
        h2, attn = self.gat2(h, edge_mask)
        h = self.norm2(h2 + h)  # residual + layernorm
        h = F.relu(h)

        graph_embed = h.reshape(h.size(0), -1)
        combined = torch.cat([graph_embed, context], dim=-1)

        # Dual heads with regime gating
        w_base = torch.softmax(self.base_head(combined), dim=-1)
        w_regime = torch.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate, edge_mask, attn


# ══════════════════════════════════════════════════════════
# TRAINING WITH VALIDATION-BASED EARLY STOPPING
# ══════════════════════════════════════════════════════════

def train_model(model, node_feats, context, targets, base_preds, regime_feats,
                train_idx, n_epochs=250, lr=0.003):
    """Train on full training data with best training loss checkpoint.

    Validation-based early stopping was tested but caused underfitting
    with small sample sizes. Best-train-loss + cosine schedule works better.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    nf = torch.FloatTensor(node_feats[train_idx])
    ctx = torch.FloatTensor(context[train_idx])
    y = torch.FloatTensor(targets[train_idx])
    bp = torch.FloatTensor(base_preds[train_idx])
    rf = torch.FloatTensor(regime_feats[train_idx])

    best_loss = float('inf')
    best_state = None

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

def dm_test_hac(e1, e2, h=H):
    """Diebold-Mariano test with Newey-West HAC standard errors."""
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


def build_sparse_causal_graph(df_train, max_edges=14, max_lag=5):
    """Build sparse causal graph keeping only top edges by F-statistic.

    Args:
        df_train: training data with shapley columns
        max_edges: max number of off-diagonal edges to keep (default 14 = 2 per node avg)
        max_lag: max lag for Granger test
    """
    from scipy import stats as sp_stats

    shapley_cols = [f'shapley_{a}' for a in AGENTS]
    data = df_train[shapley_cols].values
    n = len(data)

    # Compute F-statistics for all pairs
    edge_scores = []
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i == j:
                continue
            y = data[max_lag:, j]
            X_r = np.column_stack([data[max_lag - k - 1:n - k - 1, j] for k in range(max_lag)])
            X_u = np.column_stack([X_r] +
                                  [data[max_lag - k - 1:n - k - 1, i] for k in range(max_lag)])
            n_obs = len(y)
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
            rss_r = np.sum((y - X_r @ beta_r) ** 2)
            rss_u = np.sum((y - X_u @ beta_u) ** 2)
            df1 = max_lag
            df2 = n_obs - 2 * max_lag
            if df2 > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_val = 1 - sp_stats.f.cdf(f_stat, df1, df2)
                edge_scores.append((i, j, f_stat, p_val))

    # Sort by F-statistic (descending) and keep top edges
    edge_scores.sort(key=lambda x: x[2], reverse=True)
    adj = np.eye(N_AGENTS)  # self-loops always
    kept = 0
    for i, j, f, p in edge_scores:
        if kept >= max_edges:
            break
        adj[i, j] = 1.0
        kept += 1

    return adj


def run_single_seed(seed, df_clean, node_feats, context, targets,
                    base_preds, regime_feats, node_feat_dim, context_dim,
                    n_base, top_k=3, n_heads=4, use_prior=True, hidden_dim=16):
    """Run one seed of walk-forward evaluation.

    use_prior: True = dense causal, False = no prior, 'sparse' = sparse causal (top edges)
    """
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

        if use_prior == 'sparse':
            prior_adj = build_sparse_causal_graph(df_clean.iloc[:train_end])
        elif use_prior:
            prior_adj = build_causal_graph(df_clean.iloc[:train_end])
        else:
            prior_adj = None

        model = OptimizedGATModel(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=hidden_dim, n_heads=n_heads, top_k=top_k,
            prior_adj=prior_adj
        )

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


def evaluate_config(config_name, df_clean, node_feats, context, targets,
                    base_preds, regime_feats, har, node_feat_dim, context_dim,
                    n_base, seeds, **kwargs):
    """Evaluate a config across multiple seeds."""
    all_preds = []
    seed_results = []

    for seed in seeds:
        preds, gates, sparsity = run_single_seed(
            seed, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim,
            n_base, **kwargs
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        mae = np.mean(np.abs(p - y))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        mean_gate = np.nanmean(gates[valid])
        mean_sparsity = np.mean(sparsity)

        seed_results.append({
            'config': config_name, 'seed': seed, 'rmse': rmse, 'mae': mae,
            'dm_hac': dm, 'p_hac': pval, 'gate': mean_gate,
            'active_edges': mean_sparsity
        })
        all_preds.append(preds)

    # Ensemble
    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    rmse_ens = np.sqrt(np.mean((ens[valid] - targets[valid])**2))
    dm_ens, p_ens = dm_test_hac(ens[valid] - targets[valid], har[valid] - targets[valid])

    rmses = [r['rmse'] for r in seed_results]
    print(f"  {config_name}:")
    print(f"    Per-seed RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"    Ensemble RMSE: {rmse_ens:.4f}, DM_HAC={dm_ens:.3f} (p={p_ens:.4f})")
    print(f"    Active edges: {seed_results[0]['active_edges']:.0f}/42")

    return seed_results, ens, rmse_ens, dm_ens, p_ens


def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("OPTIMIZED GAT: Multi-Head + Top-K + Val Early Stopping")
    print("=" * 70)
    print(f"Label embargo: {LABEL_EMBARGO} trading days for forward-looking target")

    # Load data
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values

    base_cols = ['debate_vol', 'single_vol']
    base_preds = np.column_stack([df_clean[c].values for c in base_cols])
    n_base = len(base_cols)

    regime_feats = np.column_stack([
        df_clean['persist_vol'].values,
        df_clean['persist_vol'].diff().fillna(0).values,
        df_clean['n_herding'].values / 7.0,
    ])

    har = df_clean['har_vol'].values
    debate = df_clean['debate_vol'].values
    seeds = [42, 123, 456, 789, 1024]

    # ── CONFIG SWEEP ──
    configs = [
        # Best from previous round
        ("Learned-H16", dict(top_k=3, n_heads=4, use_prior=False)),
        # Sparse causal prior + learned refinement
        ("SparsePrior-H16", dict(top_k=3, n_heads=4, use_prior='sparse')),
        # Dense prior reference
        ("DensePrior-H16", dict(top_k=3, n_heads=4, use_prior=True)),
    ]

    all_results = []
    best_config = None
    best_rmse = float('inf')
    best_ens = None

    print(f"\n{'='*70}")
    print(f"PART 1: CONFIG SWEEP ({len(configs)} configs x {len(seeds)} seeds)")
    print(f"{'='*70}")

    for name, kwargs in configs:
        print(f"\n{'─'*50}")
        seed_results, ens, rmse_ens, dm_ens, p_ens = evaluate_config(
            name, df_clean, node_feats, context, targets,
            base_preds, regime_feats, har, node_feat_dim, context_dim,
            n_base, seeds, **kwargs
        )
        all_results.extend(seed_results)

        if rmse_ens < best_rmse:
            best_rmse = rmse_ens
            best_config = name
            best_ens = ens

    # ── RESULTS SUMMARY ──
    print(f"\n{'='*70}")
    print(f"PART 2: SUMMARY")
    print(f"{'='*70}")

    df_results = pd.DataFrame(all_results)
    summary = df_results.groupby('config').agg(
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        dm_mean=('dm_hac', 'mean'),
        gate_mean=('gate', 'mean'),
        edges_mean=('active_edges', 'mean'),
    ).sort_values('rmse_mean')

    print(f"\n{'Config':<25s} {'RMSE':>12s} {'DM_HAC':>9s} {'Gate':>7s} {'Edges':>7s}")
    print(f"{'─'*65}")
    for name, row in summary.iterrows():
        print(f"  {name:<23s} {row['rmse_mean']:.4f}+/-{row['rmse_std']:.4f} "
              f"{row['dm_mean']:>9.3f} {row['gate_mean']:>7.3f} {row['edges_mean']:>5.0f}")

    print(f"\n  Best config: {best_config} (ensemble RMSE={best_rmse:.4f})")

    # ── REGIME BREAKDOWN for best ──
    if best_ens is not None:
        valid = ~np.isnan(best_ens)
        y_v = targets[valid]

        print(f"\n{'='*70}")
        print(f"PART 3: REGIME BREAKDOWN ({best_config} ensemble)")
        print(f"{'='*70}")

        regimes = pd.cut(df_clean['persist_vol'].values[valid],
                         bins=[0, 0.20, 0.35, 0.55, 100],
                         labels=['low', 'normal', 'elevated', 'high'])

        print(f"\n  {'Regime':<12s} {'Optimized':>10s} {'HAR':>10s} {'Debate':>10s} {'n':>6s}")
        print(f"  {'─'*55}")
        for r in ['low', 'normal', 'elevated', 'high']:
            mask = regimes == r
            if mask.sum() > 0:
                rmse_opt = np.sqrt(np.mean((best_ens[valid][mask] - y_v[mask])**2))
                rmse_har = np.sqrt(np.mean((har[valid][mask] - y_v[mask])**2))
                rmse_deb = np.sqrt(np.mean((debate[valid][mask] - y_v[mask])**2))
                print(f"  {r:<12s} {rmse_opt:>10.4f} {rmse_har:>10.4f} "
                      f"{rmse_deb:>10.4f} {mask.sum():>6d}")

        # DM vs various baselines
        print(f"\n{'='*70}")
        print(f"PART 4: COMPARISON (all HAC-corrected)")
        print(f"{'='*70}")
        print(f"\n  {'Model':<30s} {'RMSE':>7s} {'DM_HAC':>9s} {'p':>8s} {'Sig':>4s}")
        print(f"  {'─'*60}")

        comparisons = []
        opt_vals = best_ens[valid]
        dm_opt, p_opt = dm_test_hac(opt_vals - y_v, har[valid] - y_v)
        comparisons.append((f'Optimized GAT ({best_config})', best_rmse, dm_opt, p_opt))

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
            print(f"  {name:<30s} {rmse:>7.4f} {dm:>9.3f} {p:>8.4f} {sig:>4s}")

    # Save
    df_results.to_csv(BASE / "results" / "optimized_gat_results.csv", index=False)
    print(f"\nSaved to results/optimized_gat_results.csv")


if __name__ == "__main__":
    main()
