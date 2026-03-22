"""
Post-evaluation analysis: figures + tables for the paper.
Run after run_evaluation.py produces results/.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

RESULTS_DIR = Path('results')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)


# ── Load data ─────────────────────────────────────────────

def load_all():
    wti = pd.read_csv('data/wti_full.csv', parse_dates=['date'])
    gdelt = pd.read_csv('data/gdelt_daily_features.csv', parse_dates=['date'])
    df = wti.merge(gdelt, on='date', how='left').sort_values('date').reset_index(drop=True)
    df['wti_return_fwd'] = df['wti_return'].shift(-1)
    df['realized_vol'] = df['wti_return'].rolling(20).std() * np.sqrt(252)
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


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'DA': da, 'R2_OOS': r2_oos}


# ── Table 1: Overall comparison ──────────────────────────

def table_overall_comparison():
    summary = pd.read_csv(RESULTS_DIR / 'comparison_summary.csv')
    print("\n=== TABLE 1: Overall Model Comparison ===")
    print(summary.to_string(index=False))
    return summary


# ── Figure: Agent activation timeline ────────────────────

def fig_activation_timeline(df):
    agent_df = pd.read_csv(RESULTS_DIR / 'agent_predictions.csv')
    agent_df['date'] = pd.to_datetime(agent_df['date'])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel A: WTI price + Agent regime classification
    ax = axes[0]
    dates_range = (df['date'] >= agent_df['date'].min()) & (df['date'] <= agent_df['date'].max())
    sub = df[dates_range]
    ax.plot(sub['date'], sub['wti_close'], color='#1f77b4', linewidth=0.8, alpha=0.7)

    # Overlay activated points
    activated = agent_df[agent_df['activated'] == True]
    for _, row in activated.iterrows():
        price = sub[sub['date'] == row['date']]['wti_close']
        if len(price) > 0:
            color = '#d62728' if row.get('regime') == 'crisis' else '#ff7f0e'
            ax.axvline(row['date'], color=color, alpha=0.15, linewidth=1)

    ax.set_ylabel('WTI Close ($)')
    ax.set_title('Panel A: WTI Price with Agent Activation Points')

    # Panel B: Conflict z-score + activation threshold
    ax = axes[1]
    ax.plot(sub['date'], sub['conflict_zscore'], color='#d62728', linewidth=0.8, alpha=0.7)
    ax.axhline(1.5, color='gray', linestyle='--', alpha=0.5, label='Activation threshold')
    ax.fill_between(sub['date'], sub['conflict_zscore'],
                     where=sub['conflict_zscore'] > 1.5,
                     color='#d62728', alpha=0.2)
    ax.set_ylabel('Conflict Z-Score')
    ax.set_title('Panel B: GDELT Conflict Anomaly with Activation Threshold')
    ax.legend(frameon=False)

    # Panel C: Realized vol + threshold
    ax = axes[2]
    ax.plot(sub['date'], sub['realized_vol'], color='#9467bd', linewidth=0.8, alpha=0.7)
    ax.axhline(0.40, color='gray', linestyle='--', alpha=0.5, label='Activation threshold')
    ax.fill_between(sub['date'], sub['realized_vol'],
                     where=sub['realized_vol'] > 0.40,
                     color='#9467bd', alpha=0.2)
    ax.set_ylabel('Realized Vol (ann.)')
    ax.set_title('Panel C: Realized Volatility with Activation Threshold')
    ax.legend(frameon=False)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'agent_activation_timeline.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'agent_activation_timeline.png', bbox_inches='tight')
    plt.close()
    print("Saved: figures/agent_activation_timeline.pdf")


# ── Figure: Rolling RMSE comparison ──────────────────────

def fig_rolling_rmse():
    agent_df = pd.read_csv(RESULTS_DIR / 'agent_predictions.csv')
    agent_df['date'] = pd.to_datetime(agent_df['date'])
    agent_df['sq_err'] = (agent_df['actual'] - agent_df['prediction']) ** 2

    # We need baseline predictions at the same dates
    # Approximate: use hist_mean benchmark (predict 0 = random walk)
    agent_df['sq_err_rw'] = agent_df['actual'] ** 2

    # Rolling 30-day RMSE
    window = 30
    if len(agent_df) < window:
        print("Not enough data for rolling RMSE figure")
        return

    agent_df = agent_df.sort_values('date')
    agent_df['rolling_rmse_agent'] = agent_df['sq_err'].rolling(window).mean().apply(np.sqrt)
    agent_df['rolling_rmse_rw'] = agent_df['sq_err_rw'].rolling(window).mean().apply(np.sqrt)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agent_df['date'], agent_df['rolling_rmse_agent'],
            color='#d62728', linewidth=1.2, label='Agent (LLM-orchestrated)')
    ax.plot(agent_df['date'], agent_df['rolling_rmse_rw'],
            color='#7f7f7f', linewidth=1, alpha=0.7, label='Random Walk (predict 0)')

    # Shade activated regions
    activated = agent_df[agent_df['activated'] == True]
    for _, row in activated.iterrows():
        ax.axvline(row['date'], color='#ff7f0e', alpha=0.05, linewidth=2)

    ax.set_ylabel('Rolling 30-day RMSE')
    ax.set_xlabel('Date')
    ax.set_title('Rolling RMSE: Agent vs. Random Walk Benchmark')
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rolling_rmse_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'rolling_rmse_comparison.png', bbox_inches='tight')
    plt.close()
    print("Saved: figures/rolling_rmse_comparison.pdf")


# ── Figure: Regime-conditional bar chart ─────────────────

def fig_regime_comparison(df):
    agent_df = pd.read_csv(RESULTS_DIR / 'agent_predictions.csv')
    agent_df['date'] = pd.to_datetime(agent_df['date'])

    # Merge vol info
    vol_threshold = df['realized_vol'].quantile(0.85)
    df_dates = df[['date', 'realized_vol']].copy()
    df_dates['vol_regime'] = np.where(df_dates['realized_vol'] > vol_threshold, 'High-Vol', 'Low-Vol')

    merged = agent_df.merge(df_dates[['date', 'vol_regime']], on='date', how='left')

    summary = pd.read_csv(RESULTS_DIR / 'comparison_summary.csv')

    # Agent metrics by vol regime
    regimes = ['Low-Vol', 'High-Vol']
    methods = summary['method'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, metric in enumerate(['RMSE', 'DA']):
        ax = axes[i]
        # Overall metrics from summary
        values = summary[metric].values
        x = np.arange(len(methods))
        bars = ax.barh(x, values, color=['#d62728' if 'agent' in m else '#1f77b4' for m in methods])
        ax.set_yticks(x)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel(metric)
        ax.set_title(f'Overall {metric}')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'overall_comparison_bars.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'overall_comparison_bars.png', bbox_inches='tight')
    plt.close()
    print("Saved: figures/overall_comparison_bars.pdf")


# ── Agent decision analysis ──────────────────────────────

def analyze_agent_decisions():
    try:
        with open(RESULTS_DIR / 'agent_decisions.json') as f:
            decisions = json.load(f)
    except FileNotFoundError:
        print("No agent decisions file found")
        return

    agent_df = pd.read_csv(RESULTS_DIR / 'agent_predictions.csv')

    print("\n=== AGENT DECISION ANALYSIS ===")
    print(f"Total predictions: {len(agent_df)}")
    print(f"Activated: {agent_df['activated'].sum()} ({agent_df['activated'].mean()*100:.1f}%)")

    # Performance by activation status
    for status, label in [(True, 'Activated'), (False, 'Not activated')]:
        sub = agent_df[agent_df['activated'] == status]
        if len(sub) > 0:
            m = compute_metrics(sub['actual'].values, sub['prediction'].values)
            print(f"  {label} ({len(sub)}): RMSE={m['RMSE']:.6f}, DA={m['DA']:.1f}%")

    # Performance by regime
    print("\nBy regime:")
    for regime in agent_df['regime'].unique():
        sub = agent_df[agent_df['regime'] == regime]
        if len(sub) > 5:
            m = compute_metrics(sub['actual'].values, sub['prediction'].values)
            print(f"  {regime:15s} ({len(sub):4d}): RMSE={m['RMSE']:.6f}, DA={m['DA']:.1f}%, model={sub['model_used'].iloc[0]}, feat={sub['feature_config'].iloc[0]}")

    # Drift types
    print("\nDrift types detected:")
    drift_types = [d['decision'].get('drift_type', 'none') for d in decisions if d.get('activated')]
    from collections import Counter
    for dtype, count in Counter(drift_types).most_common():
        print(f"  {dtype}: {count}")

    # Cost analysis
    agent_stats_file = RESULTS_DIR / 'agent_decisions.json'
    total_tokens = sum(1 for d in decisions if d.get('activated'))
    print(f"\nLLM calls: {total_tokens}")
    print(f"Estimated cost at $0.01/1K tokens: ${total_tokens * 1.0 * 0.01:.2f}")


# ── Iran-US case study ───────────────────────────────────

def fig_iran_case_study(df):
    agent_df = pd.read_csv(RESULTS_DIR / 'agent_predictions.csv')
    agent_df['date'] = pd.to_datetime(agent_df['date'])

    # Focus on 2025-12 to 2026-03
    iran_mask = (df['date'] >= '2025-12-01') & (df['date'] <= '2026-03-13')
    iran = df[iran_mask].copy()

    if len(iran) == 0:
        print("No Iran-US conflict period data")
        return

    agent_iran = agent_df[(agent_df['date'] >= '2025-12-01') & (agent_df['date'] <= '2026-03-13')]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel A: Price + Agent predictions
    ax = axes[0]
    ax.plot(iran['date'], iran['wti_close'], color='#1f77b4', linewidth=1.5, label='WTI Close')
    ax.set_ylabel('WTI Close ($)')
    ax.set_title('Panel A: WTI Price during Iran-US Conflict')
    ax.legend(frameon=False, loc='upper left')

    # Panel B: Agent regime classifications
    ax = axes[1]
    # Plot conflict z-score
    ax.plot(iran['date'], iran['conflict_zscore'], color='#d62728', linewidth=1.2, label='Conflict Z-Score')
    ax.axhline(1.5, color='gray', linestyle='--', alpha=0.5)

    # Mark activated points with their regime
    if len(agent_iran) > 0:
        activated = agent_iran[agent_iran['activated'] == True]
        for _, row in activated.iterrows():
            color = '#d62728' if row['regime'] == 'crisis' else '#ff7f0e'
            marker = 'D' if row['regime'] == 'crisis' else 's'
            cscore = iran[iran['date'] == row['date']]['conflict_zscore']
            if len(cscore) > 0:
                ax.scatter(row['date'], cscore.iloc[0], color=color, marker=marker, s=40, zorder=5)

    ax.set_ylabel('Conflict Z-Score')
    ax.set_title('Panel B: GDELT Conflict Signal + Agent Activation')
    ax.legend(frameon=False, loc='upper left')

    # Panel C: Agent prediction error vs baseline
    ax = axes[2]
    if len(agent_iran) > 0:
        agent_iran_sorted = agent_iran.sort_values('date')
        errors = np.abs(agent_iran_sorted['actual'] - agent_iran_sorted['prediction'])
        rw_errors = np.abs(agent_iran_sorted['actual'])  # random walk benchmark
        ax.bar(agent_iran_sorted['date'], rw_errors, width=1, alpha=0.3,
               color='#7f7f7f', label='Random Walk |error|')
        ax.bar(agent_iran_sorted['date'], errors, width=1, alpha=0.6,
               color='#d62728', label='Agent |error|')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Panel C: Agent vs. Random Walk Prediction Error')
        ax.legend(frameon=False, loc='upper left')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'iran_case_study_agent.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'iran_case_study_agent.png', bbox_inches='tight')
    plt.close()
    print("Saved: figures/iran_case_study_agent.pdf")


# ── Main ──────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_all()

    table_overall_comparison()
    analyze_agent_decisions()
    fig_activation_timeline(df)
    fig_rolling_rmse()
    fig_regime_comparison(df)
    fig_iran_case_study(df)

    print("\nAll analysis complete.")


if __name__ == "__main__":
    main()
