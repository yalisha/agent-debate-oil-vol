"""
Ablation experiments for Section 7 (Architecture Ablations).

Uses the frozen DropEdge GAT protocol:
  - 5-seed ensemble
  - 20-day label embargo
  - Walk-forward (min_train=252, retrain_every=63)
  - HAC DM test (Newey-West, Bartlett, bandwidth=19)

Ablation variants:
  A. Full model (DropEdge GAT) — re-run to export per-day predictions
  B. Graph ablations:
     1. Dense GAT (all 42 edges active, no learned sparsity)
     2. Random graph (~16/42 edges, random topology per window)
     3. Identity (self-loops only, no inter-agent edges)
  C. Feature ablations:
     4. No behaviour features (zero out behaviour + herd_streak)
     5. No Shapley/Myerson features (zero out shapley, myerson, ma5, std5)
  D. Architecture ablation:
     6. No regime gate (gate forced to 0, base_head only)

Also computes:
  - DL baseline HAC DM tests (LSTM, Transformer) from existing CSV
  - GARCH regime-conditional RMSE
  - Per-day ensemble predictions for pairwise DM tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from walk_forward_utils import embargoed_train_end, embargoed_train_idx, LABEL_EMBARGO

# ── Constants ──
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
N_AGENTS = 7
H = 20
SEEDS = [42, 123, 456, 789, 1024]
MIN_TRAIN = 252
RETRAIN_EVERY = 63

# Node feature indices (from build_features):
#   0: shapley, 1: myerson, 2: behavior/3, 3: degree/6,
#   4: shapley_ma5, 5: shapley_std5, 6: herd_streak/5
IDX_SHAPLEY = [0, 1, 4, 5]      # shapley + myerson + rolling stats
IDX_BEHAVIOR = [2, 6]            # behavior encoding + herd streak


# ══════════════════════════════════════════════════════════
# HAC DM TEST
# ══════════════════════════════════════════════════════════

def dm_test_hac(e1, e2, h=H):
    """Diebold-Mariano test with Newey-West HAC standard errors.
    e1, e2: prediction errors (pred - actual), not squared.
    """
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


# ══════════════════════════════════════════════════════════
# MODEL (imported from optimized_gat + further_optimization)
# ══════════════════════════════════════════════════════════

from optimized_gat import (
    OptimizedGATModel, MultiHeadGATLayer,
)
from further_optimization import DropEdgeGATModel, train_gat
from causal_gat_aggregation import load_data, build_features


# ── Ablation model variants ──

class DenseGATModel(DropEdgeGATModel):
    """Dense GAT: all 42 edges active, learned logits overridden."""

    def get_effective_graph(self):
        return torch.ones(N_AGENTS, N_AGENTS)


class RandomGraphGATModel(DropEdgeGATModel):
    """Random graph: ~16/42 active edges, fixed per model instance."""

    def __init__(self, *args, edge_density=16/42, graph_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        rng = np.random.RandomState(graph_seed)
        mask = np.eye(N_AGENTS)  # self-loops always
        off_diag = []
        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i != j:
                    off_diag.append((i, j))
        n_edges = int(round(edge_density * N_AGENTS * (N_AGENTS - 1)))
        chosen = rng.choice(len(off_diag), size=n_edges, replace=False)
        for idx in chosen:
            i, j = off_diag[idx]
            mask[i, j] = 1.0
        self._fixed_mask = torch.FloatTensor(mask)

    def get_effective_graph(self):
        return self._fixed_mask.to(next(self.parameters()).device)


class IdentityGATModel(DropEdgeGATModel):
    """Identity: self-loops only, no inter-agent edges."""

    def get_effective_graph(self):
        return torch.eye(N_AGENTS)


class NoRegimeGateModel(DropEdgeGATModel):
    """No regime gate: gate forced to 0, weights = base_head only."""

    def forward(self, node_feats, context, base_preds, regime_feats):
        edge_mask = self.get_effective_graph()

        if self.training and self.drop_rate > 0:
            drop_mask = (torch.rand_like(edge_mask) > self.drop_rate).float()
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

        # Base head only, gate = 0
        weights = torch.softmax(self.base_head(combined), dim=-1)
        gate = torch.zeros(node_feats.size(0), 1)

        pred = (weights * base_preds).sum(dim=-1)
        residual = self.residual_head(combined).squeeze(-1) * 0.05
        pred = torch.clamp(pred + residual, min=0.05)

        return pred, weights, gate, edge_mask, attn


# ══════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════

def run_single_seed(seed, df_clean, node_feats, context, targets,
                    base_preds, regime_feats, node_feat_dim, context_dim,
                    n_base, model_class, model_kwargs=None):
    """Run one seed of walk-forward evaluation."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_kwargs is None:
        model_kwargs = {}

    n = len(df_clean)
    preds = np.full(n, np.nan)
    retrain_points = list(range(MIN_TRAIN, n, RETRAIN_EVERY))

    for rp in retrain_points:
        chunk_end = min(rp + RETRAIN_EVERY, n)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < MIN_TRAIN:
            continue

        model = model_class(
            node_feat_dim, context_dim, n_base=n_base,
            hidden_dim=16, n_heads=4, top_k=3, prior_adj=None,
            **model_kwargs
        )

        train_gat(model, node_feats, context, targets, base_preds,
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


def evaluate_variant(name, df_clean, node_feats, context, targets,
                     base_preds, regime_feats, har, node_feat_dim,
                     context_dim, n_base, model_class, model_kwargs=None):
    """Evaluate a variant across 5 seeds with ensemble and HAC DM."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")

    all_preds = []
    seed_rmses = []

    for seed in SEEDS:
        preds = run_single_seed(
            seed, df_clean, node_feats, context, targets,
            base_preds, regime_feats, node_feat_dim, context_dim,
            n_base, model_class, model_kwargs
        )
        valid = ~np.isnan(preds)
        rmse = np.sqrt(np.mean((preds[valid] - targets[valid])**2))
        seed_rmses.append(rmse)
        all_preds.append(preds)
        print(f"    seed {seed}: RMSE={rmse:.4f}")

    # Ensemble
    ens = np.nanmean(np.stack(all_preds), axis=0)
    valid = ~np.isnan(ens)
    rmse_ens = np.sqrt(np.mean((ens[valid] - targets[valid])**2))
    dm_ens, p_ens = dm_test_hac(ens[valid] - targets[valid],
                                 har[valid] - targets[valid])

    # Regime breakdown
    def regime(v):
        if v < 0.20: return 'low'
        elif v < 0.35: return 'normal'
        elif v < 0.55: return 'elevated'
        else: return 'high'
    regimes = df_clean['persist_vol'].apply(regime).values
    regime_rmse = {}
    for r in ['low', 'normal', 'elevated', 'high']:
        mask = valid & (regimes == r)
        if mask.sum() > 0:
            regime_rmse[r] = np.sqrt(np.mean((ens[mask] - targets[mask])**2))

    print(f"  Ensemble RMSE: {rmse_ens:.4f}")
    print(f"  DM_HAC vs HAR: {dm_ens:.3f} (p={p_ens:.4f})")
    print(f"  Per-seed: {np.mean(seed_rmses):.4f} +/- {np.std(seed_rmses):.4f}")
    for r in ['low', 'normal', 'elevated', 'high']:
        if r in regime_rmse:
            print(f"    {r}: {regime_rmse[r]:.4f}")

    return {
        'name': name,
        'ensemble_rmse': rmse_ens,
        'dm_hac': dm_ens,
        'p_hac': p_ens,
        'perseed_mean': np.mean(seed_rmses),
        'perseed_std': np.std(seed_rmses),
        **{f'rmse_{r}': regime_rmse.get(r, np.nan) for r in ['low','normal','elevated','high']},
    }, ens


# ══════════════════════════════════════════════════════════
# DL BASELINE HAC DM TESTS
# ══════════════════════════════════════════════════════════

def compute_dl_baseline_dm(df_clean):
    """Compute HAC DM for LSTM and Transformer from existing CSV."""
    print(f"\n{'='*70}")
    print("DL BASELINE HAC DM TESTS")
    print(f"{'='*70}")

    dl_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                        '..', 'results', 'vol_baselines_dl_rolling.csv'))
    dl_df['date'] = pd.to_datetime(dl_df['date'])

    # Merge with clean sample to get aligned dates and HAR
    clean_dates = set(df_clean['date'].values)
    # Use aligned sample (skip first 252 training days)
    aligned = df_clean.iloc[MIN_TRAIN:].reset_index(drop=True)
    aligned_dates = set(aligned['date'].values)

    results = []
    for model_name in ['LSTM', 'Transformer', 'XGBoost']:
        sub = dl_df[dl_df['model'] == model_name].copy()
        sub['date'] = pd.to_datetime(sub['date'])
        # Merge on date with aligned sample
        merged = aligned.merge(sub[['date', 'pred_vol']], on='date',
                               how='inner', suffixes=('', '_dl'))
        if len(merged) < 100:
            print(f"  {model_name}: insufficient overlap ({len(merged)} days), skipping")
            continue

        e_dl = merged['pred_vol_dl'].values - merged['actual_vol'].values
        e_har = merged['har_vol'].values - merged['actual_vol'].values
        dm, p = dm_test_hac(e_dl, e_har)
        rmse = np.sqrt(np.mean(e_dl**2))
        print(f"  {model_name}: RMSE={rmse:.4f}, DM_HAC={dm:.3f} (p={p:.4f})")
        results.append({
            'name': model_name, 'rmse': rmse, 'dm_hac': dm, 'p_hac': p,
            'n': len(merged)
        })

    return results


# ══════════════════════════════════════════════════════════
# GARCH REGIME RMSE
# ══════════════════════════════════════════════════════════

def compute_garch_regime(df_clean):
    """Compute GARCH regime-conditional RMSE on aligned sample."""
    print(f"\n{'='*70}")
    print("GARCH REGIME-CONDITIONAL RMSE")
    print(f"{'='*70}")

    aligned = df_clean.iloc[MIN_TRAIN:].reset_index(drop=True)
    def regime(v):
        if v < 0.20: return 'low'
        elif v < 0.35: return 'normal'
        elif v < 0.55: return 'elevated'
        else: return 'high'
    aligned['regime'] = aligned['persist_vol'].apply(regime)

    e_garch = aligned['garch_vol'].values - aligned['actual_vol'].values
    e_har = aligned['har_vol'].values - aligned['actual_vol'].values
    dm, p = dm_test_hac(e_garch, e_har)
    rmse_total = np.sqrt(np.mean(e_garch**2))
    print(f"  GARCH total: RMSE={rmse_total:.4f}, DM_HAC={dm:.3f} (p={p:.4f})")

    for r in ['low', 'normal', 'elevated', 'high']:
        mask = aligned['regime'] == r
        rmse_r = np.sqrt(np.mean(e_garch[mask]**2))
        print(f"    {r} (n={mask.sum()}): RMSE={rmse_r:.4f}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ABLATION EXPERIMENTS v2 (DropEdge GAT protocol)")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Label embargo: {LABEL_EMBARGO} days")
    print(f"Walk-forward: min_train={MIN_TRAIN}, retrain_every={RETRAIN_EVERY}")

    # ── Load data ──
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values
    base_preds = np.column_stack([
        df_clean['debate_vol'].values,
        df_clean['single_vol'].values,
    ])
    n_base = 2
    regime_feats = np.column_stack([
        df_clean['persist_vol'].values,
        df_clean['persist_vol'].diff().fillna(0).values,
        df_clean['n_herding'].values / 7.0,
    ])
    har = df_clean['har_vol'].values

    # ── Define ablation variants ──
    variants = [
        # A. Full model (re-run for per-day predictions)
        ("DropEdge GAT (full)", DropEdgeGATModel,
         dict(drop_rate=0.2), node_feats),

        # B. Graph ablations
        ("Dense GAT (42/42)", DenseGATModel,
         dict(drop_rate=0.0), node_feats),
        ("Random graph (~16/42)", RandomGraphGATModel,
         dict(drop_rate=0.0, edge_density=16/42, graph_seed=42), node_feats),
        ("Identity (self-loops)", IdentityGATModel,
         dict(drop_rate=0.0), node_feats),

        # C. Feature ablations (zero out specific features)
        ("No behaviour features", DropEdgeGATModel,
         dict(drop_rate=0.2), None),  # placeholder, built below
        ("No Shapley/Myerson", DropEdgeGATModel,
         dict(drop_rate=0.2), None),  # placeholder, built below

        # D. Architecture ablation
        ("No regime gate", NoRegimeGateModel,
         dict(drop_rate=0.2), node_feats),
    ]

    # Build ablated feature arrays
    nf_no_behavior = node_feats.copy()
    nf_no_behavior[:, :, IDX_BEHAVIOR] = 0.0
    variants[4] = ("No behaviour features", DropEdgeGATModel,
                   dict(drop_rate=0.2), nf_no_behavior)

    nf_no_shapley = node_feats.copy()
    nf_no_shapley[:, :, IDX_SHAPLEY] = 0.0
    variants[5] = ("No Shapley/Myerson", DropEdgeGATModel,
                   dict(drop_rate=0.2), nf_no_shapley)

    # ── Run ablations ──
    print(f"\n{'='*70}")
    print(f"RUNNING {len(variants)} ABLATION VARIANTS x {len(SEEDS)} SEEDS")
    print(f"{'='*70}")

    all_results = []
    all_predictions = {}

    for name, model_class, model_kwargs, nf in variants:
        result, ens_preds = evaluate_variant(
            name, df_clean, nf, context, targets,
            base_preds, regime_feats, har, node_feat_dim,
            context_dim, n_base, model_class, model_kwargs
        )
        all_results.append(result)
        all_predictions[name] = ens_preds

    # ── Save results ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(results_dir, 'ablation_results_v2.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Save per-day predictions for pairwise DM tests
    pred_df = pd.DataFrame({'date': df_clean['date'].values,
                            'actual_vol': targets,
                            'har_vol': har})
    for name, preds in all_predictions.items():
        col = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        pred_df[col] = preds
    pred_path = os.path.join(results_dir, 'ablation_predictions_v2.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Per-day predictions saved to {pred_path}")

    # ── Pairwise DM matrix for key models ──
    print(f"\n{'='*70}")
    print("PAIRWISE HAC DM TESTS")
    print(f"{'='*70}")

    key_models = list(all_predictions.keys())
    # Add HAR and GARCH
    all_predictions['HAR'] = har
    all_predictions['GARCH'] = df_clean['garch_vol'].values
    all_predictions['Debate (naive)'] = df_clean['debate_vol'].values
    key_models = ['DropEdge GAT (full)', 'Dense GAT (42/42)',
                  'Identity (self-loops)', 'No regime gate',
                  'GARCH', 'HAR']

    for i, m1 in enumerate(key_models):
        for m2 in key_models[i+1:]:
            p1 = all_predictions[m1]
            p2 = all_predictions[m2]
            valid = ~(np.isnan(p1) | np.isnan(p2))
            if valid.sum() < 100:
                continue
            e1 = p1[valid] - targets[valid]
            e2 = p2[valid] - targets[valid]
            dm, p = dm_test_hac(e1, e2)
            print(f"  {m1} vs {m2}: DM_HAC={dm:.3f} (p={p:.4f})")

    # ── DL baselines ──
    dl_results = compute_dl_baseline_dm(df_clean)

    # ── GARCH regime ──
    compute_garch_regime(df_clean)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30s} {'RMSE':>8s} {'DM_HAC':>8s} {'p':>8s}")
    print("-" * 56)
    for r in all_results:
        sig = '***' if r['p_hac'] < 0.001 else '**' if r['p_hac'] < 0.01 else '*' if r['p_hac'] < 0.05 else 'n.s.'
        print(f"{r['name']:<30s} {r['ensemble_rmse']:8.4f} {r['dm_hac']:8.3f} {r['p_hac']:8.4f}  {sig}")


if __name__ == '__main__':
    main()
