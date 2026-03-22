"""
Rolling-origin evaluation: Agent vs Rule-based vs Fixed baselines.
Runs on daily WTI data with GDELT features.
"""

import pandas as pd
import numpy as np
import json
import time
import sys
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from agent_framework import (
    OilPriceAgent, AdaptivePredictor, RuleBasedPredictor,
    build_quant_snapshot, build_gdelt_summary, build_historical_context
)

# ── Config ────────────────────────────────────────────────

TRAIN_WINDOW = 252       # 1 year of trading days
TEST_START = '2023-01-01'  # Start of test period
TEST_END = '2026-03-12'
PREDICT_EVERY_N = 1       # Predict every N-th day (1 = daily, 5 = weekly)
SAVE_DIR = Path('results')
SAVE_DIR.mkdir(exist_ok=True)


# ── Load and Prepare Data ─────────────────────────────────

def prepare_data():
    """Load and merge all data sources, create features."""
    wti = pd.read_csv('data/wti_full.csv', parse_dates=['date'])
    gdelt_features = pd.read_csv('data/gdelt_daily_features.csv', parse_dates=['date'])
    gdelt_summaries = pd.read_parquet('data/gdelt_summaries/gdelt_daily_summaries.parquet')

    # Merge WTI + GDELT features
    df = wti.merge(gdelt_features, on='date', how='left')
    df = df.sort_values('date').reset_index(drop=True)

    # Create target: next-day return
    df['wti_return_fwd'] = df['wti_return'].shift(-1)

    # Create lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'wti_return_lag{lag}'] = df['wti_return'].shift(lag)

    # Realized volatility
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

    return df, gdelt_summaries


# ── Fixed Baseline Models ─────────────────────────────────

def run_fixed_baseline(df, model_name, feature_set, test_indices, train_window=252):
    """Run a fixed model with fixed features (no adaptation)."""
    results = []

    feature_map = {
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

    cols = [c for c in feature_map.get(feature_set, feature_map['minimal']) if c in df.columns]

    model_factory = {
        'ridge': lambda: Ridge(alpha=100.0),
        'gbr': lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_leaf=5, random_state=42),
        'rf': lambda: RandomForestRegressor(
            n_estimators=200, max_depth=7, max_features=0.5,
            min_samples_leaf=5, random_state=42),
        'hist_mean': None,
    }

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
            model = model_factory[model_name]()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

        results.append({
            'date': str(df.iloc[idx]['date'].date()),
            'prediction': pred,
            'actual': actual,
            'model_used': model_name,
            'feature_set': feature_set,
        })

    return pd.DataFrame(results)


# ── Evaluation Metrics ────────────────────────────────────

def compute_metrics(results_df):
    """Compute evaluation metrics."""
    y_true = results_df['actual'].values
    y_pred = results_df['prediction'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Directional accuracy
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    # R2 OOS
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)  # benchmark: predict 0
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'n_obs': len(results_df),
        'RMSE': rmse,
        'MAE': mae,
        'DA': da,
        'R2_OOS': r2_oos,
    }


def compute_regime_metrics(results_df, df):
    """Compute metrics separately for high-vol and low-vol regimes."""
    # Merge regime info
    df_dates = df[['date', 'realized_vol']].copy()
    df_dates['date'] = df_dates['date'].dt.strftime('%Y-%m-%d')
    vol_threshold = df['realized_vol'].quantile(0.85)
    df_dates['vol_regime'] = np.where(df_dates['realized_vol'] > vol_threshold, 'High-Vol', 'Low-Vol')

    # Drop 'regime' from results_df if it exists (agent results have their own regime col)
    merge_cols = results_df.drop(columns=['regime'], errors='ignore')
    merged = merge_cols.merge(df_dates[['date', 'vol_regime']], on='date', how='left')

    regime_metrics = {}
    for regime in ['Low-Vol', 'High-Vol']:
        sub = merged[merged['vol_regime'] == regime]
        if len(sub) > 10:
            regime_metrics[regime] = compute_metrics(sub)

    return regime_metrics


# ── Main Evaluation ───────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'quick'

    print("Loading data...")
    df, gdelt_summaries = prepare_data()
    print(f"Total data: {len(df)} days, {df['date'].min().date()} ~ {df['date'].max().date()}")

    # Define test period
    test_mask = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
    test_indices = df[test_mask].index.tolist()

    if mode == 'quick':
        # Quick test: every 5th day, limited period
        test_indices = test_indices[::5]
        print(f"Quick mode: {len(test_indices)} prediction points")
    elif mode == 'sample':
        # Sample 50 random points for testing
        np.random.seed(42)
        test_indices = sorted(np.random.choice(test_indices, min(50, len(test_indices)), replace=False))
        print(f"Sample mode: {len(test_indices)} prediction points")
    else:
        print(f"Full mode: {len(test_indices)} prediction points")

    all_results = {}

    # ── 1. Fixed Baselines ────────────────────────────────
    print("\n" + "="*60)
    print("FIXED BASELINES")
    print("="*60)

    baselines = [
        ('hist_mean', 'minimal'),
        ('ridge', 'minimal'),
        ('ridge', 'standard'),
        ('ridge', 'full'),
        ('gbr', 'standard'),
        ('gbr', 'full'),
        ('rf', 'standard'),
        ('rf', 'full'),
    ]

    for model_name, feat_set in baselines:
        label = f"{model_name}_{feat_set}"
        print(f"  Running {label}...", end=" ", flush=True)
        res = run_fixed_baseline(df, model_name, feat_set, test_indices)
        if len(res) > 0:
            metrics = compute_metrics(res)
            print(f"RMSE={metrics['RMSE']:.6f}, DA={metrics['DA']:.1f}%, R2={metrics['R2_OOS']:.4f}")
            all_results[label] = {'results': res, 'metrics': metrics}
        else:
            print("No results")

    # ── 2. Rule-Based Adaptive ────────────────────────────
    print("\n" + "="*60)
    print("RULE-BASED ADAPTIVE (no LLM)")
    print("="*60)

    rule_predictor = RuleBasedPredictor()
    rule_results = []

    for i, idx in enumerate(test_indices):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_indices)}", flush=True)
        try:
            res = rule_predictor.predict_one(df, idx)
            if not pd.isna(res['actual']):
                rule_results.append(res)
        except Exception as e:
            continue

    rule_df = pd.DataFrame(rule_results)
    if len(rule_df) > 0:
        rule_metrics = compute_metrics(rule_df)
        print(f"  Rule-based: RMSE={rule_metrics['RMSE']:.6f}, DA={rule_metrics['DA']:.1f}%, R2={rule_metrics['R2_OOS']:.4f}")
        all_results['rule_based'] = {'results': rule_df, 'metrics': rule_metrics}

    # ── 3. LLM Agent ─────────────────────────────────────
    print("\n" + "="*60)
    print("LLM AGENT (GPT-5.4)")
    print("="*60)

    agent = OilPriceAgent()
    adaptive_predictor = AdaptivePredictor(agent)
    agent_results = []
    agent_decisions = []

    for i, idx in enumerate(test_indices):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_indices)} (API calls: {agent.call_count}, tokens: {agent.total_tokens})", flush=True)
        try:
            res = adaptive_predictor.predict_one(df, gdelt_summaries, idx)
            if not pd.isna(res['actual']):
                agent_results.append(res)
                # Save decision log
                if agent.decision_log:
                    agent_decisions.append(agent.decision_log[-1])
        except Exception as e:
            print(f"  Error at idx {idx}: {e}")
            continue

        # Rate limiting
        if i % 3 == 0:
            time.sleep(0.5)

    agent_df = pd.DataFrame(agent_results)
    if len(agent_df) > 0:
        agent_metrics = compute_metrics(agent_df)
        print(f"  Agent: RMSE={agent_metrics['RMSE']:.6f}, DA={agent_metrics['DA']:.1f}%, R2={agent_metrics['R2_OOS']:.4f}")
        all_results['agent_gpt54'] = {'results': agent_df, 'metrics': agent_metrics}

    agent_stats = agent.get_stats()
    print(f"  Agent stats: {agent_stats}")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "="*60)
    print("OVERALL COMPARISON")
    print("="*60)

    summary_rows = []
    for label, data in all_results.items():
        row = {'method': label}
        row.update(data['metrics'])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values('RMSE')
    print(summary.to_string(index=False))

    # ── Regime-conditional metrics ────────────────────────
    print("\n" + "="*60)
    print("REGIME-CONDITIONAL METRICS")
    print("="*60)

    for label, data in all_results.items():
        regime_m = compute_regime_metrics(data['results'], df)
        for regime, metrics in regime_m.items():
            print(f"  {label:30s} | {regime:8s} | RMSE={metrics['RMSE']:.6f} DA={metrics['DA']:.1f}%")

    # ── Save results ──────────────────────────────────────
    summary.to_csv(SAVE_DIR / 'comparison_summary.csv', index=False)

    if len(agent_df) > 0:
        agent_df.to_csv(SAVE_DIR / 'agent_predictions.csv', index=False)

    # Save agent decisions
    if agent_decisions:
        with open(SAVE_DIR / 'agent_decisions.json', 'w') as f:
            json.dump(agent_decisions, f, indent=2, default=str)

    print(f"\nResults saved to {SAVE_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
