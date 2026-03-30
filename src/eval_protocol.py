"""
P0: Rigorous evaluation protocol.
- Fixed seeds, 3-seed mean/std
- HAC/Newey-West DM test (bandwidth = h-1 = 19 for h=20 target)
- Proper date-sorted walk-forward
- Autocorrelation diagnostics on forecast errors
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from scipy import stats
from walk_forward_utils import LABEL_EMBARGO, embargoed_train_end, embargoed_train_idx

BASE = Path(__file__).parent.parent
AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
H = 20  # forecast horizon for HAC bandwidth


# ══════════════════════════════════════════════════════════
# 1. HAC DM TEST
# ══════════════════════════════════════════════════════════

def dm_test_hac(e1, e2, h=H):
    """
    Diebold-Mariano test with Newey-West HAC standard errors.

    Args:
        e1, e2: prediction errors (not squared) from two models
        h: forecast horizon (determines HAC bandwidth = h-1)

    Returns:
        dm_stat, p_value, d_bar, hac_se
    """
    d = e1**2 - e2**2
    d = d[~np.isnan(d)]
    T = len(d)
    if T < h + 10:
        return np.nan, np.nan, np.nan, np.nan

    d_bar = d.mean()

    # Newey-West HAC variance with Bartlett kernel
    # bandwidth = h - 1 for h-step-ahead forecasts
    bandwidth = h - 1
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, bandwidth + 1):
        if k < T:
            weight = 1.0 - k / (bandwidth + 1)  # Bartlett kernel
            cov_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            gamma_sum += weight * cov_k

    var_d = (gamma_0 + 2 * gamma_sum) / T

    # Guard against negative variance estimate
    if var_d <= 0:
        var_d = gamma_0 / T

    hac_se = np.sqrt(var_d)
    dm_stat = d_bar / hac_se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value, d_bar, hac_se


def dm_test_naive(e1, e2):
    """Standard DM test without HAC (for comparison)."""
    d = e1**2 - e2**2
    d = d[~np.isnan(d)]
    T = len(d)
    d_bar = d.mean()
    se = d.std(ddof=1) / np.sqrt(T)
    if se <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / se
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# ══════════════════════════════════════════════════════════
# 2. AUTOCORRELATION DIAGNOSTICS
# ══════════════════════════════════════════════════════════

def error_autocorrelation(errors, max_lag=25):
    """Compute autocorrelation of forecast errors to diagnose overlap."""
    e = errors[~np.isnan(errors)]
    n = len(e)
    e_centered = e - e.mean()
    var = np.var(e_centered)
    if var == 0:
        return {}

    acf = {}
    for k in range(1, min(max_lag + 1, n)):
        acf[k] = np.mean(e_centered[k:] * e_centered[:-k]) / var
    return acf


# ══════════════════════════════════════════════════════════
# 3. MULTI-SEED GAT EVALUATION
# ══════════════════════════════════════════════════════════

def run_gat_single_seed(seed, df_clean, node_feats, context, targets,
                        base_preds, base_cols, graph_mode='causal'):
    """Run GAT with a specific random seed."""
    from causal_gat_aggregation import (
        CausalGATAggregator, build_causal_graph,
    )
    from run_ablations import make_full_graph, make_identity_graph

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(df_clean)
    min_train = 252
    retrain_every = 63
    n_base = len(base_cols)
    node_feat_dim = node_feats.shape[2]
    context_dim = context.shape[1]

    preds = np.full(n, np.nan)
    retrain_points = list(range(min_train, n, retrain_every))

    for rp in retrain_points:
        chunk_end = min(rp + retrain_every, n)
        train_end = embargoed_train_end(rp)
        train_idx = embargoed_train_idx(rp)
        test_idx = np.arange(rp, chunk_end)
        if len(train_idx) < min_train:
            continue

        # Graph
        if graph_mode == 'causal':
            adj = build_causal_graph(df_clean.iloc[:train_end])
        elif graph_mode == 'full':
            adj = make_full_graph()
        else:
            adj = np.eye(len(AGENTS))

        # Model
        model = CausalGATAggregator(node_feat_dim, context_dim, hidden_dim=16)
        if n_base != 5:
            import torch.nn as nn
            model.N_BASE = n_base
            model.weight_head = nn.Sequential(
                nn.Linear(len(AGENTS) * 16 + context_dim, 32),
                nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, n_base),
            )

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        loss_fn = torch.nn.MSELoss()

        nf_t = torch.FloatTensor(node_feats[train_idx])
        ctx_t = torch.FloatTensor(context[train_idx])
        y_t = torch.FloatTensor(targets[train_idx])
        bp_t = torch.FloatTensor(base_preds[train_idx])

        best_loss, best_state = float('inf'), None
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            pred, _, _ = model(nf_t, adj, ctx_t, bp_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        # Predict
        model.eval()
        with torch.no_grad():
            nf_test = torch.FloatTensor(node_feats[test_idx])
            ctx_test = torch.FloatTensor(context[test_idx])
            bp_test = torch.FloatTensor(base_preds[test_idx])
            pred, _, _ = model(nf_test, adj, ctx_test, bp_test)
            pred = torch.clamp(pred, min=0.05)
        preds[test_idx] = pred.numpy()

    return preds


def main():
    print("=" * 70)
    print("P0: RIGOROUS EVALUATION PROTOCOL")
    print("=" * 70)
    print(f"Label embargo: {LABEL_EMBARGO} trading days for forward-looking target")

    # Load data, sort by date
    from causal_gat_aggregation import load_data, build_features
    df = load_data()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    clean_mask = (df['actual_vol'] <= 2) & (df['persist_vol'] <= 2)
    df_clean = df[clean_mask].reset_index(drop=True)
    dates = pd.to_datetime(df_clean['date'])
    assert dates.is_monotonic_increasing, "Data not sorted by date!"
    print(f"Date range: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
    print(f"Clean sample: n={len(df_clean)}")

    node_feats, context, node_feat_dim, context_dim = build_features(df_clean)
    targets = df_clean['actual_vol'].values
    n = len(df_clean)

    # Walk-forward eval range
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[252:] = True
    n_eval = eval_mask.sum()
    y = targets[eval_mask]

    # ──────────────────────────────────────────────────────
    # Part 1: Autocorrelation diagnostics
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 1: FORECAST ERROR AUTOCORRELATION DIAGNOSTICS")
    print(f"{'─'*70}")

    for name, col in [('Debate', 'debate_vol'), ('HAR', 'har_vol'),
                      ('Persistence', 'persist_vol')]:
        errors = df_clean[col].values[eval_mask] - y
        acf = error_autocorrelation(errors)
        sig_lags = [k for k, v in acf.items() if abs(v) > 1.96 / np.sqrt(n_eval)]
        print(f"\n  {name} error ACF:")
        print(f"    lag 1:  {acf.get(1, 0):.3f}")
        print(f"    lag 5:  {acf.get(5, 0):.3f}")
        print(f"    lag 10: {acf.get(10, 0):.3f}")
        print(f"    lag 19: {acf.get(19, 0):.3f}")
        print(f"    lag 20: {acf.get(20, 0):.3f}")
        print(f"    Significant lags (|ACF| > 1.96/sqrt(n)): {sig_lags[:10]}")

    # ──────────────────────────────────────────────────────
    # Part 2: Naive vs HAC DM tests on existing results
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 2: NAIVE vs HAC DM TESTS (existing model outputs)")
    print(f"{'─'*70}")

    har_errors = df_clean['har_vol'].values[eval_mask] - y

    print(f"\n  {'Model':<25s} {'DM_naive':>10s} {'p_naive':>8s} {'DM_HAC':>10s} {'p_HAC':>8s} {'Shrinkage':>10s}")
    print(f"  {'─'*75}")

    for name, col in [('Debate', 'debate_vol'), ('Single', 'single_vol'),
                      ('Persistence', 'persist_vol'), ('GARCH', 'garch_vol')]:
        errors = df_clean[col].values[eval_mask] - y
        dm_n, p_n = dm_test_naive(errors, har_errors)
        dm_h, p_h, _, _ = dm_test_hac(errors, har_errors)
        shrink = (1 - abs(dm_h) / abs(dm_n)) * 100 if not np.isnan(dm_n) and dm_n != 0 else 0
        sig_n = '*' if p_n < 0.05 else ''
        sig_h = '*' if p_h < 0.05 else ''
        print(f"  {name:<25s} {dm_n:>9.3f}{sig_n} {p_n:>8.4f} {dm_h:>9.3f}{sig_h} {p_h:>8.4f} {shrink:>9.1f}%")

    # GAT results (if available)
    gat_path = BASE / "results" / "causal_gat_results.csv"
    if gat_path.exists():
        gat_df = pd.read_csv(gat_path)
        gat_df['date'] = pd.to_datetime(gat_df['date'])
        merged = df_clean.merge(gat_df[['date', 'gat_pred']], on='date', how='inner')
        gat_errors = merged['gat_pred'].values - merged['actual_vol'].values
        har_errors_gat = merged['har_vol'].values - merged['actual_vol'].values

        dm_n, p_n = dm_test_naive(gat_errors, har_errors_gat)
        dm_h, p_h, _, _ = dm_test_hac(gat_errors, har_errors_gat)
        shrink = (1 - abs(dm_h) / abs(dm_n)) * 100 if dm_n != 0 else 0
        sig_n = '*' if p_n < 0.05 else ''
        sig_h = '*' if p_h < 0.05 else ''
        print(f"  {'Causal-GAT (saved)':<25s} {dm_n:>9.3f}{sig_n} {p_n:>8.4f} {dm_h:>9.3f}{sig_h} {p_h:>8.4f} {shrink:>9.1f}%")

    # ──────────────────────────────────────────────────────
    # Part 3: Multi-seed GAT evaluation
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 3: MULTI-SEED EVALUATION (3 seeds)")
    print(f"{'─'*70}")

    seeds = [42, 123, 456]

    # Key variants to test with multi-seed
    all_base = ['debate_vol', 'har_vol', 'persist_vol', 'single_vol', 'garch_vol']
    llm_base = ['debate_vol', 'single_vol']

    configs = [
        ("Causal-GAT (all 5)", all_base, 'causal'),
        ("Causal-GAT (LLM only)", llm_base, 'causal'),
    ]

    for config_name, base_cols, graph_mode in configs:
        print(f"\n  Config: {config_name}")
        base_preds = np.column_stack([df_clean[c].values for c in base_cols])

        seed_rmses = []
        seed_maes = []
        seed_dms_naive = []
        seed_dms_hac = []
        all_preds = []

        for seed in seeds:
            print(f"    Seed {seed}...", end=" ", flush=True)
            preds = run_gat_single_seed(
                seed, df_clean, node_feats, context, targets,
                base_preds, base_cols, graph_mode
            )

            valid = ~np.isnan(preds)
            p = preds[valid]
            y_valid = targets[valid]
            har_valid = df_clean['har_vol'].values[valid]

            rmse = np.sqrt(np.mean((p - y_valid)**2))
            mae = np.mean(np.abs(p - y_valid))

            errors = p - y_valid
            har_e = har_valid - y_valid
            dm_n, _ = dm_test_naive(errors, har_e)
            dm_h, p_h, _, _ = dm_test_hac(errors, har_e)

            seed_rmses.append(rmse)
            seed_maes.append(mae)
            seed_dms_naive.append(dm_n)
            seed_dms_hac.append(dm_h)
            all_preds.append(preds)

            print(f"RMSE={rmse:.4f}, DM_HAC={dm_h:.3f} (p={p_h:.4f})")

        mean_rmse = np.mean(seed_rmses)
        std_rmse = np.std(seed_rmses)
        mean_dm_hac = np.mean(seed_dms_hac)
        std_dm_hac = np.std(seed_dms_hac)
        mean_dm_naive = np.mean(seed_dms_naive)

        print(f"    ────────────────────────────")
        print(f"    RMSE:     {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"    DM_naive: {mean_dm_naive:.3f} ± {np.std(seed_dms_naive):.3f}")
        print(f"    DM_HAC:   {mean_dm_hac:.3f} ± {std_dm_hac:.3f}")
        print(f"    HAC shrinkage: {(1 - abs(mean_dm_hac)/abs(mean_dm_naive))*100:.1f}%")

        # Ensemble of 3 seeds (average predictions)
        ensemble_preds = np.nanmean(np.stack(all_preds), axis=0)
        valid = ~np.isnan(ensemble_preds)
        p_ens = ensemble_preds[valid]
        y_ens = targets[valid]
        har_ens = df_clean['har_vol'].values[valid]

        rmse_ens = np.sqrt(np.mean((p_ens - y_ens)**2))
        dm_h_ens, p_h_ens, _, _ = dm_test_hac(p_ens - y_ens, har_ens - y_ens)
        print(f"    Ensemble (avg 3 seeds): RMSE={rmse_ens:.4f}, DM_HAC={dm_h_ens:.3f} (p={p_h_ens:.4f})")

    # ──────────────────────────────────────────────────────
    # Part 4: Baselines with HAC DM
    # ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PART 4: ALL BASELINES WITH HAC DM TEST")
    print(f"{'─'*70}")

    # ML baselines
    bl_path = BASE / "results" / "vol_baselines_full.csv"
    dl_path = BASE / "results" / "vol_baselines_dl_rolling.csv"

    print(f"\n  {'Model':<20s} {'RMSE':>7s} {'MAE':>7s} {'DM_HAC':>9s} {'p_HAC':>8s} {'Sig':>4s} {'n':>5s}")
    print(f"  {'─'*65}")

    # Built-in models (debate period)
    har_e_full = df_clean['har_vol'].values - targets
    for name, col in [('Debate', 'debate_vol'), ('Single', 'single_vol'),
                      ('Persistence', 'persist_vol'), ('HAR', 'har_vol'),
                      ('GARCH', 'garch_vol')]:
        vals = df_clean[col].values
        rmse = np.sqrt(np.mean((vals - targets)**2))
        mae = np.mean(np.abs(vals - targets))
        if name != 'HAR':
            errors = vals - targets
            dm_h, p_h, _, _ = dm_test_hac(errors, har_e_full)
            sig = '*' if p_h < 0.05 else ''
        else:
            dm_h, p_h, sig = 0, 1, ''
        print(f"  {name:<20s} {rmse:>7.4f} {mae:>7.4f} {dm_h:>9.3f} {p_h:>8.4f} {sig:>4s} {len(targets):>5d}")

    # ML baselines
    if bl_path.exists():
        bl = pd.read_csv(bl_path, parse_dates=['date'])
        bl_clean = bl[(bl['actual_vol'] <= 2) & (bl['persist_vol'] <= 2)]
        for model_name in ['Ridge', 'Lasso', 'GBR', 'RF']:
            sub = bl_clean[bl_clean['model'] == model_name]
            if len(sub) == 0:
                continue
            merged = df_clean[['date', 'har_vol', 'actual_vol']].merge(
                sub[['date', 'pred_vol']], on='date')
            errors = merged['pred_vol'].values - merged['actual_vol'].values
            har_e = merged['har_vol'].values - merged['actual_vol'].values
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            dm_h, p_h, _, _ = dm_test_hac(errors, har_e)
            sig = '*' if p_h < 0.05 else ''
            print(f"  {model_name:<20s} {rmse:>7.4f} {mae:>7.4f} {dm_h:>9.3f} {p_h:>8.4f} {sig:>4s} {len(merged):>5d}")

    # DL baselines
    if dl_path.exists():
        dl = pd.read_csv(dl_path, parse_dates=['date'])
        dl_clean = dl[(dl['actual_vol'] <= 2) & (dl['wti_vol_20d'] <= 2)]
        for model_name in ['XGBoost', 'LSTM', 'Transformer']:
            sub = dl_clean[dl_clean['model'] == model_name]
            if len(sub) == 0:
                continue
            merged = df_clean[['date', 'har_vol', 'actual_vol']].merge(
                sub[['date', 'pred_vol']], on='date')
            errors = merged['pred_vol'].values - merged['actual_vol'].values
            har_e = merged['har_vol'].values - merged['actual_vol'].values
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            dm_h, p_h, _, _ = dm_test_hac(errors, har_e)
            sig = '*' if p_h < 0.05 else ''
            print(f"  {model_name:<20s} {rmse:>7.4f} {mae:>7.4f} {dm_h:>9.3f} {p_h:>8.4f} {sig:>4s} {len(merged):>5d}")

    print(f"\n  Note: DM_HAC uses Newey-West with Bartlett kernel, bandwidth={H-1}")
    print(f"  * = significant at 5% level")


if __name__ == "__main__":
    main()
