"""
Optimized GAT v2: Enhanced leak-free features.

Experiment matrix:
  A. Lagged Shapley (lag=20): baseline, already shown ~0.1520
  B. Rich leak-free features (no Shapley at all): behaviour dynamics,
     rolling agent patterns, debate system diagnostics
  C. Lagged Shapley + rich features combined

The goal is to find leak-free node features that allow the GAT to
differentiate agents and outperform the MLP/HAR baselines.

Run: cd /Users/mac/computerscience/17Agent可解释预测 && /opt/miniconda3/bin/python -u src/optimized_gat_v2.py
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
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════

def _add_common_derived(df):
    """Add derived columns used by all feature builders."""
    for agent in AGENTS:
        herd_ind = (df[f'behavior_{agent}'] == 'herding').astype(float)
        df[f'herd_streak_{agent}'] = herd_ind.rolling(5, min_periods=1).sum()
    if 'vol_change_5d' not in df.columns:
        df['vol_change_5d'] = df['persist_vol'].diff(5).fillna(0)
    if 'debate_har_gap' not in df.columns:
        df['debate_har_gap'] = df['debate_vol'] - df['har_vol']


def _build_context(df):
    """Build context features (same across all variants, all leak-free)."""
    context = np.column_stack([
        df['persist_vol'].values,
        df['har_vol'].values,
        df['total_adj'].values,
        df['n_herding'].values / 7.0,
        df['vol_regime'].values / 3.0,
        df['debate_vol'].values,
        (df['persist_vol'] - df['har_vol']).values,
        df['vol_change_5d'].values,
        df['debate_har_gap'].values,
    ])
    return context


def build_features_lagged(df):
    """Variant A: Lagged Shapley/Myerson (lag=20). Already tested."""
    _add_common_derived(df)
    lag = LABEL_EMBARGO
    for agent in AGENTS:
        df[f'shapley_lag_{agent}'] = df[f'shapley_{agent}'].shift(lag).fillna(0)
        df[f'myerson_lag_{agent}'] = df[f'myerson_{agent}'].shift(lag).fillna(0)
        df[f'shapley_ma5_lag_{agent}'] = df[f'shapley_lag_{agent}'].rolling(5, min_periods=1).mean()
        df[f'shapley_std5_lag_{agent}'] = df[f'shapley_lag_{agent}'].rolling(5, min_periods=1).std().fillna(0)

    node_feat_dim = 7
    node_feats = np.zeros((len(df), N_AGENTS, node_feat_dim))
    for i, agent in enumerate(AGENTS):
        node_feats[:, i, 0] = df[f'shapley_lag_{agent}'].values
        node_feats[:, i, 1] = df[f'myerson_lag_{agent}'].values
        node_feats[:, i, 2] = df[f'beh_enc_{agent}'].values / 3.0
        node_feats[:, i, 3] = df.get(f'degree_{agent}', pd.Series(0, index=df.index)).values / 6.0
        node_feats[:, i, 4] = df[f'shapley_ma5_lag_{agent}'].values
        node_feats[:, i, 5] = df[f'shapley_std5_lag_{agent}'].values
        node_feats[:, i, 6] = df[f'herd_streak_{agent}'].values / 5.0

    context = _build_context(df)
    return node_feats, context, node_feat_dim, context.shape[1]


def build_features_rich(df):
    """Variant B: Rich leak-free features (NO Shapley at all).

    Per-agent features (all available at prediction time t):
    0. beh_enc / 3                   - current behaviour type
    1. degree / 6                    - in-degree in debate influence graph
    2. herd_streak / 5               - consecutive herding days (last 5)
    3. rolling_herd_freq_10          - fraction of last 10 days this agent herded
    4. behaviour_diversity_10        - number of distinct behaviour types in last 10 days / 4
    5. is_independent                - 1 if current behaviour is independent, 0 otherwise
    6. degree_ma5 / 6                - 5-day rolling mean of degree (persistence of influence)
    7. herd_minus_independent_10     - (herding days - independent days) / 10 in last 10 window
    """
    _add_common_derived(df)

    for agent in AGENTS:
        herd_ind = (df[f'behavior_{agent}'] == 'herding').astype(float)
        indep_ind = (df[f'behavior_{agent}'] == 'independent').astype(float)

        # Rolling herding frequency (last 10 days)
        df[f'herd_freq10_{agent}'] = herd_ind.rolling(10, min_periods=1).mean()

        # Behaviour diversity: how many distinct types in last 10 days
        beh_enc = df[f'beh_enc_{agent}']
        df[f'beh_div10_{agent}'] = beh_enc.rolling(10, min_periods=1).apply(
            lambda x: len(set(x)) / 4.0, raw=True
        )

        # Is independent indicator
        df[f'is_indep_{agent}'] = indep_ind

        # Rolling degree
        deg_col = f'degree_{agent}'
        if deg_col in df.columns:
            df[f'degree_ma5_{agent}'] = df[deg_col].rolling(5, min_periods=1).mean()
        else:
            df[f'degree_ma5_{agent}'] = 0.0

        # Herding - independent balance
        df[f'herd_indep_bal_{agent}'] = (
            herd_ind.rolling(10, min_periods=1).sum() -
            indep_ind.rolling(10, min_periods=1).sum()
        ) / 10.0

    node_feat_dim = 8
    node_feats = np.zeros((len(df), N_AGENTS, node_feat_dim))
    for i, agent in enumerate(AGENTS):
        node_feats[:, i, 0] = df[f'beh_enc_{agent}'].values / 3.0
        node_feats[:, i, 1] = df.get(f'degree_{agent}', pd.Series(0, index=df.index)).values / 6.0
        node_feats[:, i, 2] = df[f'herd_streak_{agent}'].values / 5.0
        node_feats[:, i, 3] = df[f'herd_freq10_{agent}'].values
        node_feats[:, i, 4] = df[f'beh_div10_{agent}'].values
        node_feats[:, i, 5] = df[f'is_indep_{agent}'].values
        node_feats[:, i, 6] = df[f'degree_ma5_{agent}'].values / 6.0
        node_feats[:, i, 7] = df[f'herd_indep_bal_{agent}'].values

    context = _build_context(df)
    return node_feats, context, node_feat_dim, context.shape[1]


def build_features_combined(df):
    """Variant C: Lagged Shapley + rich behaviour features combined.

    Per-agent features (11-dim):
    0. shapley_lag (t-20)
    1. myerson_lag (t-20)
    2. beh_enc / 3
    3. degree / 6
    4. herd_streak / 5
    5. shapley_ma5_lag
    6. shapley_std5_lag
    7. herd_freq10 (rolling 10-day herding frequency)
    8. beh_div10 (behaviour diversity)
    9. is_independent
    10. herd_indep_bal (herding minus independent balance)
    """
    _add_common_derived(df)
    lag = LABEL_EMBARGO

    for agent in AGENTS:
        # Lagged Shapley
        df[f'shapley_lag_{agent}'] = df[f'shapley_{agent}'].shift(lag).fillna(0)
        df[f'myerson_lag_{agent}'] = df[f'myerson_{agent}'].shift(lag).fillna(0)
        df[f'shapley_ma5_lag_{agent}'] = df[f'shapley_lag_{agent}'].rolling(5, min_periods=1).mean()
        df[f'shapley_std5_lag_{agent}'] = df[f'shapley_lag_{agent}'].rolling(5, min_periods=1).std().fillna(0)

        # Rich behaviour features
        herd_ind = (df[f'behavior_{agent}'] == 'herding').astype(float)
        indep_ind = (df[f'behavior_{agent}'] == 'independent').astype(float)
        df[f'herd_freq10_{agent}'] = herd_ind.rolling(10, min_periods=1).mean()
        beh_enc = df[f'beh_enc_{agent}']
        df[f'beh_div10_{agent}'] = beh_enc.rolling(10, min_periods=1).apply(
            lambda x: len(set(x)) / 4.0, raw=True
        )
        df[f'is_indep_{agent}'] = indep_ind
        df[f'herd_indep_bal_{agent}'] = (
            herd_ind.rolling(10, min_periods=1).sum() -
            indep_ind.rolling(10, min_periods=1).sum()
        ) / 10.0

    node_feat_dim = 11
    node_feats = np.zeros((len(df), N_AGENTS, node_feat_dim))
    for i, agent in enumerate(AGENTS):
        node_feats[:, i, 0] = df[f'shapley_lag_{agent}'].values
        node_feats[:, i, 1] = df[f'myerson_lag_{agent}'].values
        node_feats[:, i, 2] = df[f'beh_enc_{agent}'].values / 3.0
        node_feats[:, i, 3] = df.get(f'degree_{agent}', pd.Series(0, index=df.index)).values / 6.0
        node_feats[:, i, 4] = df[f'herd_streak_{agent}'].values / 5.0
        node_feats[:, i, 5] = df[f'shapley_ma5_lag_{agent}'].values
        node_feats[:, i, 6] = df[f'shapley_std5_lag_{agent}'].values
        node_feats[:, i, 7] = df[f'herd_freq10_{agent}'].values
        node_feats[:, i, 8] = df[f'beh_div10_{agent}'].values
        node_feats[:, i, 9] = df[f'is_indep_{agent}'].values
        node_feats[:, i, 10] = df[f'herd_indep_bal_{agent}'].values

    context = _build_context(df)
    return node_feats, context, node_feat_dim, context.shape[1]


# ══════════════════════════════════════════════════════════
# AUGMENTED CONTEXT: add system-level leak-free diagnostics
# ══════════════════════════════════════════════════════════

def augment_context_with_diagnostics(df, context):
    """Add system-level diagnostic features to context vector.

    New features (all leak-free):
    - rolling_debate_accuracy_lag20: |debate_vol(t-20) - actual_vol(t-20)|
      By day t, actual_vol(t-20) is fully observed.
    - rolling_n_herding_ma5: 5-day trend in herding count
    - debate_single_gap: debate_vol - single_vol (debate effect)
    """
    lag = LABEL_EMBARGO

    # Rolling debate accuracy (lagged)
    debate_err_lag = np.abs(
        df['debate_vol'].shift(lag).values - df['actual_vol'].shift(lag).values
    )
    debate_err_lag = np.nan_to_num(debate_err_lag, nan=0.0)
    debate_acc_roll = pd.Series(debate_err_lag).rolling(5, min_periods=1).mean().values

    # Herding trend
    n_herd_ma5 = df['n_herding'].rolling(5, min_periods=1).mean().values / 7.0

    # Debate vs single gap
    debate_single_gap = (df['debate_vol'] - df['single_vol']).values

    aug = np.column_stack([
        context,
        debate_acc_roll,
        n_herd_ma5,
        debate_single_gap,
    ])
    return aug


# ══════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════

from optimized_gat import (
    OptimizedGATModel,
    dm_test_hac,
)
from further_optimization import (
    DropEdgeGATModel,
    run_gat_seed,
    run_mlp_seed,
    evaluate_method,
)
from causal_gat_aggregation import load_data


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GAT v2: LEAK-FREE FEATURE EXPERIMENTS")
    print("=" * 70)

    # Load and clean data
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    n = len(df_clean)
    print(f"Clean sample: n={n}")

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
    persist_vol = df_clean['persist_vol'].values
    seeds = [42, 123, 456, 789, 1024]

    all_results = []

    # ══════════════════════════════════════════════════════
    # Variant A: Lagged Shapley (reference, already tested)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VARIANT A: Lagged Shapley (reference)")
    print(f"{'='*70}")

    df_a = df_clean.copy()
    nf_a, ctx_a, nfd_a, ctxd_a = build_features_lagged(df_a)
    ctx_a_aug = augment_context_with_diagnostics(df_a, ctx_a)

    # DropEdge GAT
    r = evaluate_method(
        "A-DropEdge-LagShapley", run_gat_seed,
        df_a, nf_a, ctx_a_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_a, ctx_a_aug.shape[1], n_base, seeds,
        model_cls=DropEdgeGATModel, drop_rate=0.2
    )
    all_results.append(r)

    # MLP
    r = evaluate_method(
        "A-MLP-LagShapley", run_mlp_seed,
        df_a, nf_a, ctx_a_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_a, ctx_a_aug.shape[1], n_base, seeds,
    )
    all_results.append(r)

    # ══════════════════════════════════════════════════════
    # Variant B: Rich behaviour features (no Shapley)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VARIANT B: Rich behaviour features (no Shapley)")
    print(f"{'='*70}")

    df_b = df_clean.copy()
    nf_b, ctx_b, nfd_b, ctxd_b = build_features_rich(df_b)
    ctx_b_aug = augment_context_with_diagnostics(df_b, ctx_b)

    r = evaluate_method(
        "B-DropEdge-RichBeh", run_gat_seed,
        df_b, nf_b, ctx_b_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_b, ctx_b_aug.shape[1], n_base, seeds,
        model_cls=DropEdgeGATModel, drop_rate=0.2
    )
    all_results.append(r)

    r = evaluate_method(
        "B-MLP-RichBeh", run_mlp_seed,
        df_b, nf_b, ctx_b_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_b, ctx_b_aug.shape[1], n_base, seeds,
    )
    all_results.append(r)

    # ══════════════════════════════════════════════════════
    # Variant C: Lagged Shapley + Rich combined
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VARIANT C: Lagged Shapley + Rich combined")
    print(f"{'='*70}")

    df_c = df_clean.copy()
    nf_c, ctx_c, nfd_c, ctxd_c = build_features_combined(df_c)
    ctx_c_aug = augment_context_with_diagnostics(df_c, ctx_c)

    r = evaluate_method(
        "C-DropEdge-Combined", run_gat_seed,
        df_c, nf_c, ctx_c_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_c, ctx_c_aug.shape[1], n_base, seeds,
        model_cls=DropEdgeGATModel, drop_rate=0.2
    )
    all_results.append(r)

    r = evaluate_method(
        "C-MLP-Combined", run_mlp_seed,
        df_c, nf_c, ctx_c_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_c, ctx_c_aug.shape[1], n_base, seeds,
    )
    all_results.append(r)

    # ══════════════════════════════════════════════════════
    # Variant D: Minimal (behaviour only, 3 dims)
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VARIANT D: Minimal behaviour features (3-dim)")
    print(f"{'='*70}")

    df_d = df_clean.copy()
    _add_common_derived(df_d)
    nfd_d = 3
    nf_d = np.zeros((len(df_d), N_AGENTS, nfd_d))
    for i, agent in enumerate(AGENTS):
        nf_d[:, i, 0] = df_d[f'beh_enc_{agent}'].values / 3.0
        nf_d[:, i, 1] = df_d.get(f'degree_{agent}', pd.Series(0, index=df_d.index)).values / 6.0
        nf_d[:, i, 2] = df_d[f'herd_streak_{agent}'].values / 5.0
    ctx_d = _build_context(df_d)
    ctx_d_aug = augment_context_with_diagnostics(df_d, ctx_d)

    r = evaluate_method(
        "D-DropEdge-Minimal", run_gat_seed,
        df_d, nf_d, ctx_d_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_d, ctx_d_aug.shape[1], n_base, seeds,
        model_cls=DropEdgeGATModel, drop_rate=0.2
    )
    all_results.append(r)

    r = evaluate_method(
        "D-MLP-Minimal", run_mlp_seed,
        df_d, nf_d, ctx_d_aug, targets, base_preds, regime_feats,
        har, persist_vol, nfd_d, ctx_d_aug.shape[1], n_base, seeds,
    )
    all_results.append(r)

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("FULL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30s} {'RMSE':>8s} {'DM_HAC':>8s} {'p':>8s} {'Sig':>5s}")
    print(f"{'─'*65}")
    print(f"  {'v1 DropEdge (LEAK)':<28s} {'0.1171':>8s} {'-3.212':>8s} {'0.0013':>8s} {'***':>5s}")
    print(f"  {'HAR benchmark':<28s} {'0.1549':>8s} {'---':>8s} {'---':>8s} {'---':>5s}")
    for r in all_results:
        sig = '***' if r['p_hac'] < 0.001 else ('**' if r['p_hac'] < 0.01 else
              ('*' if r['p_hac'] < 0.05 else ('.' if r['p_hac'] < 0.1 else 'n.s.')))
        print(f"  {r['name']:<28s} {r['ensemble_rmse']:>8.4f} {r['dm_hac']:>8.3f} {r['p_hac']:>8.4f} {sig:>5s}")

    # Pairwise GAT vs MLP for each variant
    print(f"\n{'='*70}")
    print("GAT vs MLP COMPARISONS")
    print(f"{'='*70}")
    variants = ['A', 'B', 'C', 'D']
    for v in variants:
        gat = [r for r in all_results if r['name'].startswith(f'{v}-DropEdge')][0]
        mlp = [r for r in all_results if r['name'].startswith(f'{v}-MLP')][0]
        gp, mp = gat['ensemble_preds'], mlp['ensemble_preds']
        valid = ~np.isnan(gp) & ~np.isnan(mp)
        if valid.sum() > 0:
            dm, p = dm_test_hac(gp[valid] - targets[valid], mp[valid] - targets[valid])
            print(f"  Variant {v}: GAT={gat['ensemble_rmse']:.4f} vs MLP={mlp['ensemble_rmse']:.4f}, "
                  f"DM_HAC={dm:.3f} (p={p:.4f})")

    # Save
    df_out = pd.DataFrame([{
        'name': r['name'], 'ensemble_rmse': r['ensemble_rmse'],
        'dm_hac': r['dm_hac'], 'p_hac': r['p_hac'],
        'perseed_mean': r['perseed_mean'], 'perseed_std': r['perseed_std'],
        **{f'rmse_{k}': r.get(f'rmse_{k}', np.nan) for k in ['low', 'normal', 'elevated', 'high']},
    } for r in all_results])
    out_path = BASE / 'results' / 'optimized_gat_v2_feature_experiments.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
