"""
Base model tuning with time-series cross-validation.
Goal: find optimal hyperparameters for each model × feature_set × regime combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = Path('results')
SAVE_DIR.mkdir(exist_ok=True)

# ── Data Loading (same as run_evaluation.py) ──────────────

def prepare_data():
    wti = pd.read_csv('data/wti_full.csv', parse_dates=['date'])
    gdelt_features = pd.read_csv('data/gdelt_daily_features.csv', parse_dates=['date'])

    df = wti.merge(gdelt_features, on='date', how='left')
    df = df.sort_values('date').reset_index(drop=True)

    # Target: next-day return
    df['wti_return_fwd'] = df['wti_return'].shift(-1)

    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'wti_return_lag{lag}'] = df['wti_return'].shift(lag)

    # Realized vol
    df['realized_vol'] = df['wti_return'].rolling(20).std() * np.sqrt(252)

    # Conflict features
    if 'oil_material_conflict_share' in df.columns:
        df['conflict_intensity'] = (
            df['oil_material_conflict_share'].fillna(0) * 0.4 +
            df['oil_conflict_share'].fillna(0) * 0.3 +
            (-df['oil_goldstein_mean'].fillna(0) / 10).clip(0, 1) * 0.3
        )
        df['conflict_intensity_7d'] = df['conflict_intensity'].rolling(7).mean()
        df['conflict_zscore'] = (
            (df['conflict_intensity_7d'] - df['conflict_intensity_7d'].rolling(60).mean()) /
            df['conflict_intensity_7d'].rolling(60).std()
        )

    return df


# ── Feature Sets ──────────────────────────────────────────

def get_feature_cols(df, feature_set):
    cols_map = {
        'minimal': [c for c in df.columns if c.startswith('wti_return_lag')],
        'standard': [c for c in df.columns if c.startswith('wti_return_lag')
                     or c in ['VIX', 'DXY', 'YIELD_10Y', 'realized_vol']],
        'full': [c for c in df.columns if c.startswith('wti_return_lag')
                 or c in ['VIX', 'DXY', 'YIELD_10Y', 'realized_vol',
                          'conflict_intensity_7d', 'conflict_zscore',
                          'oil_material_conflict_share', 'oil_goldstein_mean',
                          'oil_conflict_share', 'oil_tone_mean',
                          'oil_net_coop']],
    }
    return [c for c in cols_map.get(feature_set, cols_map['minimal']) if c in df.columns]


# ── Model Configurations ────────────────────────────────

MODEL_CONFIGS = {
    # Ridge with different alphas
    'ridge_a0.1': lambda: Ridge(alpha=0.1),
    'ridge_a1': lambda: Ridge(alpha=1.0),
    'ridge_a10': lambda: Ridge(alpha=10.0),
    'ridge_a100': lambda: Ridge(alpha=100.0),
    'ridge_a1000': lambda: Ridge(alpha=1000.0),

    # Auto-tuned Ridge
    'ridgecv': lambda: RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),

    # Lasso (automatic feature selection)
    'lassocv': lambda: LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0], cv=5, max_iter=5000),

    # ElasticNet
    'elasticnetcv': lambda: ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9], alphas=[0.0001, 0.001, 0.01, 0.1],
        cv=5, max_iter=5000
    ),

    # GBR variants
    'gbr_conservative': lambda: GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.01,
        subsample=0.8, min_samples_leaf=20, random_state=42
    ),
    'gbr_moderate': lambda: GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    ),
    'gbr_aggressive': lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42
    ),

    # RF variants
    'rf_conservative': lambda: RandomForestRegressor(
        n_estimators=100, max_depth=3, max_features='sqrt',
        min_samples_leaf=20, random_state=42
    ),
    'rf_moderate': lambda: RandomForestRegressor(
        n_estimators=200, max_depth=5, max_features='sqrt',
        min_samples_leaf=10, random_state=42
    ),
    'rf_aggressive': lambda: RandomForestRegressor(
        n_estimators=200, max_depth=7, max_features=0.5,
        min_samples_leaf=5, random_state=42
    ),

    # Historical mean (benchmark)
    'hist_mean': None,
}


# ── Evaluation ──────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'DA': da, 'R2_OOS': r2_oos}


def rolling_origin_eval(df, model_name, feature_set, test_indices, train_window=252):
    """Evaluate one model config on given test indices."""
    cols = get_feature_cols(df, feature_set)
    if not cols:
        return []

    results = []
    for idx in test_indices:
        train_start = max(0, idx - train_window)
        train = df.iloc[train_start:idx].dropna(subset=cols + ['wti_return_fwd'])

        if len(train) < 30:
            continue

        actual = df.iloc[idx].get('wti_return_fwd', np.nan)
        if pd.isna(actual):
            continue

        if model_name == 'hist_mean':
            pred = train['wti_return_fwd'].mean()
        else:
            X_train = train[cols].fillna(0)
            y_train = train['wti_return_fwd']
            X_test = df.iloc[[idx]][cols].fillna(0)
            model = MODEL_CONFIGS[model_name]()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

        # Regime info for regime-conditional analysis
        vol = df.iloc[idx].get('realized_vol', 0)
        if pd.isna(vol):
            vol = 0

        results.append({
            'date': str(df.iloc[idx]['date'].date()),
            'prediction': pred,
            'actual': actual,
            'realized_vol': vol,
        })

    return results


def main():
    print("Loading data...")
    df = prepare_data()

    # Test period
    test_mask = (df['date'] >= '2023-01-01') & (df['date'] <= '2026-03-12')
    all_test_indices = df[test_mask].index.tolist()

    # Use every 5th day for tuning speed (still 150+ points)
    test_indices = all_test_indices[::5]
    print(f"Tuning on {len(test_indices)} points (every 5th day)")

    # Volatility threshold for regime split
    vol_threshold = df['realized_vol'].quantile(0.85)
    print(f"Vol threshold (85th pct): {vol_threshold:.4f}")

    # ── Run all model × feature_set combinations ────────
    all_metrics = []
    feature_sets = ['minimal', 'standard', 'full']

    for feat_set in feature_sets:
        cols = get_feature_cols(df, feat_set)
        print(f"\n{'='*60}")
        print(f"Feature set: {feat_set} ({len(cols)} features)")
        print(f"  Features: {cols}")
        print(f"{'='*60}")

        for model_name in MODEL_CONFIGS:
            print(f"  {model_name:25s}", end=" ", flush=True)

            results = rolling_origin_eval(df, model_name, feat_set, test_indices)
            if not results:
                print("No results")
                continue

            res_df = pd.DataFrame(results)

            # Overall metrics
            overall = compute_metrics(res_df['actual'].values, res_df['prediction'].values)
            print(f"RMSE={overall['RMSE']:.6f}  DA={overall['DA']:.1f}%  R2={overall['R2_OOS']:.4f}", end="")

            row = {
                'model': model_name,
                'feature_set': feat_set,
                'n_obs': len(res_df),
                **{f'overall_{k}': v for k, v in overall.items()},
            }

            # Low-vol regime
            low_vol = res_df[res_df['realized_vol'] <= vol_threshold]
            if len(low_vol) > 10:
                lv_m = compute_metrics(low_vol['actual'].values, low_vol['prediction'].values)
                row.update({f'lowvol_{k}': v for k, v in lv_m.items()})

            # High-vol regime
            high_vol = res_df[res_df['realized_vol'] > vol_threshold]
            if len(high_vol) > 5:
                hv_m = compute_metrics(high_vol['actual'].values, high_vol['prediction'].values)
                row.update({f'highvol_{k}': v for k, v in hv_m.items()})
                print(f"  | HV: RMSE={hv_m['RMSE']:.6f} DA={hv_m['DA']:.1f}%", end="")

            print()
            all_metrics.append(row)

    # ── Summary ─────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    results_df = results_df.sort_values('overall_RMSE')

    print("\n" + "="*80)
    print("TOP 10 OVERALL (sorted by RMSE)")
    print("="*80)
    top_cols = ['model', 'feature_set', 'n_obs', 'overall_RMSE', 'overall_DA', 'overall_R2_OOS']
    print(results_df[top_cols].head(10).to_string(index=False))

    # Best per regime
    if 'lowvol_RMSE' in results_df.columns:
        print("\n" + "="*80)
        print("TOP 5 LOW-VOL REGIME")
        print("="*80)
        lv_cols = ['model', 'feature_set', 'lowvol_RMSE', 'lowvol_DA', 'lowvol_R2_OOS']
        valid = results_df.dropna(subset=['lowvol_RMSE']).sort_values('lowvol_RMSE')
        print(valid[lv_cols].head(5).to_string(index=False))

    if 'highvol_RMSE' in results_df.columns:
        print("\n" + "="*80)
        print("TOP 5 HIGH-VOL REGIME")
        print("="*80)
        hv_cols = ['model', 'feature_set', 'highvol_RMSE', 'highvol_DA', 'highvol_R2_OOS']
        valid = results_df.dropna(subset=['highvol_RMSE']).sort_values('highvol_RMSE')
        print(valid[hv_cols].head(5).to_string(index=False))

    # ── Best model per regime ────────────────────────────
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION")
    print("="*80)

    if 'lowvol_RMSE' in results_df.columns:
        best_lv = results_df.dropna(subset=['lowvol_RMSE']).sort_values('lowvol_RMSE').iloc[0]
        print(f"  Low-Vol:  {best_lv['model']} + {best_lv['feature_set']} (RMSE={best_lv['lowvol_RMSE']:.6f})")

    if 'highvol_RMSE' in results_df.columns:
        best_hv = results_df.dropna(subset=['highvol_RMSE']).sort_values('highvol_RMSE').iloc[0]
        print(f"  High-Vol: {best_hv['model']} + {best_hv['feature_set']} (RMSE={best_hv['highvol_RMSE']:.6f})")

    best_overall = results_df.iloc[0]
    print(f"  Overall:  {best_overall['model']} + {best_overall['feature_set']} (RMSE={best_overall['overall_RMSE']:.6f})")

    # Save
    results_df.to_csv(SAVE_DIR / 'model_tuning_results.csv', index=False)
    print(f"\nFull results saved to {SAVE_DIR / 'model_tuning_results.csv'}")


if __name__ == "__main__":
    main()
