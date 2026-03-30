"""
Final comparison under unified protocol:
1. Same-protocol no-graph baseline (MLP with regime head, 5-seed ensemble, HAC DM)
2. Edge stability analysis for learned graph (across seeds and walk-forward windows)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from collections import defaultdict
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = len(AGENTS)
H = 20


# ══════════════════════════════════════════════════════════
# NO-GRAPH BASELINE (same regime-gated head, same features)
# ══════════════════════════════════════════════════════════

class OptimizedMLPModel(nn.Module):
    """No-graph baseline with same output structure as OptimizedGAT.

    Uses flattened node features + context (no message passing).
    Same regime-gated dual head for fair comparison.
    """
    def __init__(self, node_feat_dim, context_dim, n_base=2, hidden_dim=16):
        super().__init__()
        self.n_base = n_base

        input_dim = N_AGENTS * node_feat_dim + context_dim

        # Feature encoder (replaces GAT)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        graph_dim = hidden_dim + context_dim

        # Same dual head structure as OptimizedGAT
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

    def forward(self, node_feats, context, base_preds, regime_feats):
        flat = node_feats.reshape(node_feats.size(0), -1)
        combined_in = torch.cat([flat, context], dim=-1)
        encoded = self.encoder(combined_in)

        combined = torch.cat([encoded, context], dim=-1)

        w_base = torch.softmax(self.base_head(combined), dim=-1)
        w_regime = torch.softmax(self.regime_head(combined), dim=-1)
        gate = self.regime_gate(regime_feats)
        weights = (1 - gate) * w_base + gate * w_regime

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate


# ══════════════════════════════════════════════════════════
# OPTIMIZED GAT (imported architecture)
# ══════════════════════════════════════════════════════════

from optimized_gat import (
    OptimizedGATModel, MultiHeadGATLayer,
    dm_test_hac
)


# ══════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════

def train_mlp(model, node_feats, context, targets, base_preds, regime_feats,
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
        pred, _, _ = model(nf, ctx, bp, rf)
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
# WALK-FORWARD WITH EDGE TRACKING
# ══════════════════════════════════════════════════════════

def run_mlp_seed(seed, df_clean, node_feats, context, targets,
                 base_preds, regime_feats, node_feat_dim, context_dim, n_base):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)
    all_gates = np.full(n, np.nan)

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
            pred, weights, gate = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()
        all_gates[test_idx] = gate.squeeze(-1).numpy()

    return preds, all_gates


def run_gat_seed_with_edges(seed, df_clean, node_feats, context, targets,
                            base_preds, regime_feats, node_feat_dim, context_dim, n_base):
    """Run GAT and record learned edge structure at each retrain window."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    preds = np.full(n, np.nan)
    all_gates = np.full(n, np.nan)
    edge_snapshots = []  # list of (window_idx, edge_logits_matrix)

    retrain_points = list(range(252, n, 63))

    for wi, rp in enumerate(retrain_points):
        chunk_end = min(rp + 63, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < 252:
            continue

        model = OptimizedGATModel(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=16, n_heads=4, top_k=3, prior_adj=None
        )
        train_gat(model, node_feats, context, targets, base_preds,
                  regime_feats, train_idx)

        # Record edge structure
        with torch.no_grad():
            edge_gates = torch.sigmoid(model.edge_logits).numpy()
            edge_snapshots.append({
                'window': wi,
                'train_end': rp,
                'edge_gates': edge_gates.copy(),
                'active_edges': ((edge_gates * (1 - np.eye(N_AGENTS))) > 0.5).sum()
            })

        model.eval()
        with torch.no_grad():
            nf = torch.FloatTensor(node_feats[test_idx])
            ctx = torch.FloatTensor(context[test_idx])
            bp = torch.FloatTensor(base_preds[test_idx])
            rf = torch.FloatTensor(regime_feats[test_idx])
            pred, weights, gate, _, _ = model(nf, ctx, bp, rf)

        preds[test_idx] = pred.numpy()
        all_gates[test_idx] = gate.squeeze(-1).numpy()

    return preds, all_gates, edge_snapshots


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    from causal_gat_aggregation import load_data, build_features

    print("=" * 70)
    print("FINAL COMPARISON: Learned GAT vs No-Graph Baseline")
    print("Unified protocol: 5-seed ensemble, HAC DM test")
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

    # ════════════════════════════════════════════
    # PART 1: NO-GRAPH BASELINE
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 1: OPTIMIZED MLP (NO GRAPH) BASELINE")
    print(f"{'='*70}")

    mlp_preds_all = []
    mlp_results = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        preds, gates = run_mlp_seed(
            seed, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim, n_base
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        print(f"RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={pval:.4f})")
        mlp_results.append({'seed': seed, 'rmse': rmse, 'dm_hac': dm, 'p_hac': pval})
        mlp_preds_all.append(preds)

    mlp_ens = np.nanmean(np.stack(mlp_preds_all), axis=0)
    valid = ~np.isnan(mlp_ens)
    mlp_rmse_ens = np.sqrt(np.mean((mlp_ens[valid] - targets[valid])**2))
    mlp_dm_ens, mlp_p_ens = dm_test_hac(mlp_ens[valid] - targets[valid], har[valid] - targets[valid])

    mlp_rmses = [r['rmse'] for r in mlp_results]
    print(f"\n  MLP Summary:")
    print(f"    Per-seed: {np.mean(mlp_rmses):.4f} +/- {np.std(mlp_rmses):.4f}")
    print(f"    Ensemble: {mlp_rmse_ens:.4f}, DM_HAC={mlp_dm_ens:.3f} (p={mlp_p_ens:.4f})")

    # ════════════════════════════════════════════
    # PART 2: LEARNED GAT WITH EDGE TRACKING
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 2: LEARNED GAT WITH EDGE STABILITY ANALYSIS")
    print(f"{'='*70}")

    gat_preds_all = []
    gat_results = []
    all_edge_snapshots = {}  # seed -> list of snapshots

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        preds, gates, edge_snapshots = run_gat_seed_with_edges(
            seed, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim, n_base
        )
        valid = ~np.isnan(preds)
        y = targets[valid]
        p = preds[valid]
        rmse = np.sqrt(np.mean((p - y)**2))
        dm, pval = dm_test_hac(p - y, har[valid] - y)
        print(f"RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={pval:.4f}), "
              f"avg_edges={np.mean([s['active_edges'] for s in edge_snapshots]):.0f}/42")
        gat_results.append({'seed': seed, 'rmse': rmse, 'dm_hac': dm, 'p_hac': pval})
        gat_preds_all.append(preds)
        all_edge_snapshots[seed] = edge_snapshots

    gat_ens = np.nanmean(np.stack(gat_preds_all), axis=0)
    valid = ~np.isnan(gat_ens)
    gat_rmse_ens = np.sqrt(np.mean((gat_ens[valid] - targets[valid])**2))
    gat_dm_ens, gat_p_ens = dm_test_hac(gat_ens[valid] - targets[valid], har[valid] - targets[valid])

    gat_rmses = [r['rmse'] for r in gat_results]
    print(f"\n  GAT Summary:")
    print(f"    Per-seed: {np.mean(gat_rmses):.4f} +/- {np.std(gat_rmses):.4f}")
    print(f"    Ensemble: {gat_rmse_ens:.4f}, DM_HAC={gat_dm_ens:.3f} (p={gat_p_ens:.4f})")

    # DM test: GAT vs MLP
    dm_gat_mlp, p_gat_mlp = dm_test_hac(
        gat_ens[valid] - targets[valid],
        mlp_ens[valid] - targets[valid]
    )

    # ════════════════════════════════════════════
    # PART 3: HEAD-TO-HEAD COMPARISON
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 3: HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")

    y_v = targets[valid]
    print(f"\n  {'Model':<30s} {'RMSE':>7s} {'DM_HAC':>9s} {'p':>8s} {'Sig':>4s}")
    print(f"  {'─'*60}")

    comparisons = [
        ('Learned GAT (ensemble)', gat_rmse_ens, gat_dm_ens, gat_p_ens),
        ('MLP no-graph (ensemble)', mlp_rmse_ens, mlp_dm_ens, mlp_p_ens),
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
        print(f"  {name:<30s} {rmse:>7.4f} {dm:>9.3f} {p:>8.4f} {sig:>4s}")

    print(f"\n  GAT vs MLP (direct): DM_HAC={dm_gat_mlp:.3f} (p={p_gat_mlp:.4f})"
          f" {'*' if p_gat_mlp < 0.05 else 'n.s.'}")

    # Regime breakdown
    print(f"\n  {'Regime':<12s} {'GAT':>8s} {'MLP':>8s} {'HAR':>8s} {'Debate':>8s} {'n':>6s}")
    print(f"  {'─'*50}")
    regimes = pd.cut(df_clean['persist_vol'].values[valid],
                     bins=[0, 0.20, 0.35, 0.55, 100],
                     labels=['low', 'normal', 'elevated', 'high'])
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = regimes == r
        if mask.sum() > 0:
            r_gat = np.sqrt(np.mean((gat_ens[valid][mask] - y_v[mask])**2))
            r_mlp = np.sqrt(np.mean((mlp_ens[valid][mask] - y_v[mask])**2))
            r_har = np.sqrt(np.mean((har[valid][mask] - y_v[mask])**2))
            r_deb = np.sqrt(np.mean((debate[valid][mask] - y_v[mask])**2))
            print(f"  {r:<12s} {r_gat:>8.4f} {r_mlp:>8.4f} {r_har:>8.4f} {r_deb:>8.4f} {mask.sum():>6d}")

    # ════════════════════════════════════════════
    # PART 4: EDGE STABILITY ANALYSIS
    # ════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 4: EDGE STABILITY ANALYSIS")
    print(f"{'='*70}")

    n_windows = len(list(all_edge_snapshots.values())[0])
    n_seeds = len(seeds)

    # 4a. Edge frequency across seeds (at each window)
    # For each (i,j), count how many (seed, window) pairs have edge active
    edge_freq = np.zeros((N_AGENTS, N_AGENTS))
    total_counts = n_seeds * n_windows

    for seed, snapshots in all_edge_snapshots.items():
        for snap in snapshots:
            gates = snap['edge_gates']
            active = ((gates * (1 - np.eye(N_AGENTS))) > 0.5).astype(float)
            edge_freq += active

    edge_freq_pct = edge_freq / total_counts * 100

    # 4b. Cross-seed stability: for each window, which edges appear in all seeds
    window_stability = []
    for wi in range(n_windows):
        # Get edge matrices across seeds for this window
        seed_edges = []
        for seed in seeds:
            gates = all_edge_snapshots[seed][wi]['edge_gates']
            active = ((gates * (1 - np.eye(N_AGENTS))) > 0.5).astype(float)
            seed_edges.append(active)
        # Count per-edge agreement
        agreement = np.mean(seed_edges, axis=0)  # fraction of seeds that have this edge
        n_unanimous = (agreement == 1.0).sum()  # edges present in ALL seeds
        n_majority = (agreement >= 0.6).sum()   # edges present in >= 60% of seeds
        window_stability.append({
            'window': wi,
            'train_end': all_edge_snapshots[seeds[0]][wi]['train_end'],
            'unanimous': n_unanimous,
            'majority': n_majority,
        })

    print(f"\n  4a. Edge frequency across all ({n_seeds} seeds x {n_windows} windows):")
    print(f"  {'Edge':<35s} {'Freq%':>7s} {'Classification':>15s}")
    print(f"  {'─'*60}")

    edge_list = []
    for i in range(N_AGENTS):
        for j in range(N_AGENTS):
            if i == j:
                continue
            freq = edge_freq_pct[i, j]
            edge_list.append((AGENTS[i], AGENTS[j], freq))

    edge_list.sort(key=lambda x: x[2], reverse=True)

    for src, dst, freq in edge_list:
        if freq > 80:
            cls = "STABLE"
        elif freq > 50:
            cls = "FREQUENT"
        elif freq > 20:
            cls = "OCCASIONAL"
        else:
            cls = "RARE"
        if freq > 10:  # only show non-trivial
            print(f"  {src:>15s} -> {dst:<15s} {freq:>6.1f}%   {cls:>15s}")

    # 4c. Temporal stability
    print(f"\n  4b. Cross-seed agreement per walk-forward window:")
    print(f"  {'Window':>8s} {'TrainEnd':>10s} {'Unanimous':>11s} {'Majority':>10s}")
    print(f"  {'─'*45}")
    for ws in window_stability:
        print(f"  {ws['window']:>8d} {ws['train_end']:>10d} {ws['unanimous']:>11d} {ws['majority']:>10d}")

    # 4d. Top stable edges summary
    stable_edges = [(s, d, f) for s, d, f in edge_list if f > 50]
    rare_edges = [(s, d, f) for s, d, f in edge_list if f < 10]
    print(f"\n  4c. Summary:")
    print(f"    Stable edges (>50%): {len(stable_edges)}/42")
    print(f"    Rare edges (<10%): {len(rare_edges)}/42")
    print(f"    Average active edges per model: {edge_freq.sum() / total_counts:.1f}")

    # 4e. Regime-specific edge patterns
    print(f"\n  4d. Edge structure in early vs late training windows:")
    early_windows = [0, 1, 2]
    late_windows = [n_windows-3, n_windows-2, n_windows-1]

    early_freq = np.zeros((N_AGENTS, N_AGENTS))
    late_freq = np.zeros((N_AGENTS, N_AGENTS))
    for seed in seeds:
        for wi in early_windows:
            if wi < len(all_edge_snapshots[seed]):
                gates = all_edge_snapshots[seed][wi]['edge_gates']
                early_freq += ((gates * (1 - np.eye(N_AGENTS))) > 0.5).astype(float)
        for wi in late_windows:
            if wi < len(all_edge_snapshots[seed]):
                gates = all_edge_snapshots[seed][wi]['edge_gates']
                late_freq += ((gates * (1 - np.eye(N_AGENTS))) > 0.5).astype(float)

    early_pct = early_freq / (n_seeds * len(early_windows)) * 100
    late_pct = late_freq / (n_seeds * len(late_windows)) * 100

    print(f"\n  {'Edge':<35s} {'Early%':>8s} {'Late%':>8s} {'Shift':>8s}")
    print(f"  {'─'*65}")
    for src, dst, freq in edge_list[:20]:  # top 20
        e = early_pct[AGENTS.index(src), AGENTS.index(dst)]
        l = late_pct[AGENTS.index(src), AGENTS.index(dst)]
        shift = l - e
        if abs(shift) > 10:
            marker = " <<<"
        else:
            marker = ""
        print(f"  {src:>15s} -> {dst:<15s} {e:>7.1f}% {l:>7.1f}% {shift:>+7.1f}%{marker}")

    # Save all results
    results = {
        'mlp': mlp_results,
        'gat': gat_results,
        'mlp_ensemble_rmse': mlp_rmse_ens,
        'gat_ensemble_rmse': gat_rmse_ens,
        'gat_vs_mlp_dm': dm_gat_mlp,
        'gat_vs_mlp_p': p_gat_mlp,
    }

    # Save edge frequency matrix
    edge_df = pd.DataFrame(edge_freq_pct, index=AGENTS, columns=AGENTS)
    edge_df.to_csv(BASE / "results" / "edge_frequency_matrix.csv")

    # Save comparison results
    comp_df = pd.DataFrame({
        'model': ['Learned GAT', 'MLP no-graph'],
        'ensemble_rmse': [gat_rmse_ens, mlp_rmse_ens],
        'dm_hac_vs_har': [gat_dm_ens, mlp_dm_ens],
        'p_hac_vs_har': [gat_p_ens, mlp_p_ens],
        'perseed_mean': [np.mean(gat_rmses), np.mean(mlp_rmses)],
        'perseed_std': [np.std(gat_rmses), np.std(mlp_rmses)],
    })
    comp_df.to_csv(BASE / "results" / "final_comparison.csv", index=False)

    print(f"\nSaved edge_frequency_matrix.csv and final_comparison.csv")


if __name__ == "__main__":
    main()
