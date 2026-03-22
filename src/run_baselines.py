"""
Run baseline models on daily WTI return forecasting with rolling-origin evaluation.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from evaluation import RollingOriginEvaluator, compute_metrics
from baseline_models import get_baseline_models

DATA_DIR = "/Users/mac/computerscience/17Agent可解释预测/data"
RESULTS_DIR = "/Users/mac/computerscience/17Agent可解释预测/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_features(oil_path, gdelt_path, regime_path=None):
    """
    Merge oil + GDELT data and construct features for daily return prediction.
    All features use only information available at time t to predict return at t+h.
    """
    oil = pd.read_csv(oil_path, parse_dates=['date'])
    gdelt = pd.read_csv(gdelt_path, parse_dates=['date'])

    df = oil.merge(gdelt, on='date', how='left')
    df = df.sort_values('date').reset_index(drop=True)

    # Target: next-day return (shifted forward, so we predict r_{t+1} using info at t)
    df['target_r1'] = df['wti_return'].shift(-1)
    # 5-day forward return
    df['target_r5'] = df['wti_price'].pct_change(5).shift(-5)

    # === Feature engineering (all using info available at time t) ===

    # Lagged returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag_{lag}'] = df['wti_return'].shift(lag)

    # Realized volatility features
    df['vol_5d'] = df['wti_return'].rolling(5).std()
    df['vol_10d'] = df['wti_return'].rolling(10).std()
    df['vol_ratio'] = df['vol_5d'] / (df['vol_10d'] + 1e-8)

    # Price momentum
    df['mom_5d'] = df['wti_price'].pct_change(5)
    df['mom_20d'] = df['wti_price'].pct_change(20)

    # Macro features (already in the data, just ensure they're lagged properly)
    # These are same-day values which is OK since they're market-observable
    macro_features = ['VIX', 'DXY', 'YIELD_SPREAD_10Y_2Y', 'REAL_YIELD_10Y']
    for col in macro_features:
        if col in df.columns:
            df[f'{col}_change'] = df[col].diff()
            df[f'{col}_change_5d'] = df[col].diff(5)

    # GDELT features (already daily, lagged by 1 to avoid look-ahead)
    gdelt_features = [
        'oil_goldstein_mean', 'oil_conflict_share',
        'oil_material_conflict_share', 'oil_net_coop',
        'oil_tone_mean', 'oil_mentions_sum',
        'n_events_oil', 'global_goldstein_mean',
    ]
    for col in gdelt_features:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            # 5-day moving average
            df[f'{col}_ma5'] = df[col].rolling(5).mean().shift(1)

    return df


def get_feature_sets():
    """Define different feature sets for ablation."""

    base_features = [
        'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10', 'ret_lag_20',
        'vol_5d', 'vol_10d', 'vol_ratio',
        'mom_5d', 'mom_20d',
    ]

    macro_features = [
        'VIX_change', 'DXY_change', 'YIELD_SPREAD_10Y_2Y_change',
        'REAL_YIELD_10Y_change',
        'VIX_change_5d', 'DXY_change_5d',
    ]

    gdelt_features = [
        'oil_goldstein_mean_lag1', 'oil_conflict_share_lag1',
        'oil_material_conflict_share_lag1', 'oil_net_coop_lag1',
        'oil_tone_mean_lag1', 'oil_mentions_sum_lag1',
        'n_events_oil_lag1',
        'oil_goldstein_mean_ma5', 'oil_conflict_share_ma5',
        'oil_tone_mean_ma5',
    ]

    return {
        'returns_only': base_features,
        'returns_macro': base_features + macro_features,
        'full': base_features + macro_features + gdelt_features,
    }


def main():
    print("=" * 60)
    print("BASELINE EVALUATION: Daily WTI Return Forecasting")
    print("=" * 60)

    # Prepare data
    oil_path = f"{DATA_DIR}/oil_macro_daily.csv"
    gdelt_path = f"{DATA_DIR}/gdelt_daily_features.csv"

    df = prepare_features(oil_path, gdelt_path)
    print(f"Total observations: {len(df)}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Get models and feature sets
    models = get_baseline_models()
    feature_sets = get_feature_sets()

    # Evaluator: 2-year training window, daily step=5 to keep runtime reasonable
    evaluator = RollingOriginEvaluator(
        train_window=504,   # ~2 years
        horizons=[1, 5],    # 1-day and 5-day ahead
        step=5,             # evaluate every 5 days
        min_train=504,
    )

    all_results = []

    for fs_name, features in feature_sets.items():
        # Filter to features that actually exist
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"\n[{fs_name}] Missing features (skipped): {missing}")

        print(f"\n{'='*60}")
        print(f"Feature set: {fs_name} ({len(available)} features)")
        print(f"{'='*60}")

        results = evaluator.run(
            df=df,
            features=available,
            target='target_r1',
            models=models,
            regime_col='regime' if 'regime' in df.columns else None,
        )

        for r in results:
            r.metadata['feature_set'] = fs_name

        all_results.extend(results)

        # Quick summary
        metrics = compute_metrics(results, group_by=['model_name', 'horizon'])
        print(f"\nResults for {fs_name}:")
        print(metrics[['model_name', 'horizon', 'n_obs', 'RMSE', 'MAE', 'DA', 'R2_OOS']]
              .to_string(index=False))

    # Overall comparison
    print(f"\n{'='*60}")
    print("OVERALL COMPARISON")
    print(f"{'='*60}")

    results_df = pd.DataFrame([
        {**vars(r), 'feature_set': r.metadata.get('feature_set', '')}
        for r in all_results
    ])

    overall = compute_metrics(
        all_results,
        group_by=['model_name', 'horizon']
    )
    overall_sorted = overall.sort_values(['horizon', 'RMSE'])
    print("\nBest models by horizon (sorted by RMSE):")
    for h in [1, 5]:
        print(f"\n  Horizon = {h}:")
        sub = overall_sorted[overall_sorted['horizon'] == h].head(8)
        print(sub[['model_name', 'n_obs', 'RMSE', 'MAE', 'DA', 'R2_OOS']].to_string(index=False))

    # Save detailed results
    results_df.to_csv(f"{RESULTS_DIR}/baseline_detailed_results.csv", index=False)
    overall_sorted.to_csv(f"{RESULTS_DIR}/baseline_summary.csv", index=False)

    # Regime-conditional metrics (if regime is available)
    if 'regime' in results_df.columns and results_df['regime'].notna().any():
        print(f"\n{'='*60}")
        print("REGIME-CONDITIONAL METRICS")
        print(f"{'='*60}")

        regime_metrics = compute_metrics(
            all_results,
            group_by=['model_name', 'horizon', 'regime']
        )
        regime_metrics = regime_metrics.sort_values(['horizon', 'regime', 'RMSE'])
        regime_metrics.to_csv(f"{RESULTS_DIR}/baseline_regime_metrics.csv", index=False)

        for h in [1]:
            for r in [0, 1]:
                label = 'Low-Vol' if r == 0 else 'High-Vol'
                sub = regime_metrics[(regime_metrics['horizon'] == h) & (regime_metrics['regime'] == r)]
                if not sub.empty:
                    print(f"\n  Horizon={h}, Regime={label}:")
                    print(sub[['model_name', 'n_obs', 'RMSE', 'MAE', 'DA']].head(5).to_string(index=False))

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
