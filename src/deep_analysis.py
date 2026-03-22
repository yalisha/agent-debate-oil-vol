#!/usr/bin/env /opt/miniconda3/bin/python
"""
Deep analysis for multi-agent LLM debate system.
6 modules: case studies, behavioral regressions, intervention fix,
Shapley significance, cascade dynamics, master comparison.
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, kruskal, ttest_1samp
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
DOCS = BASE / 'docs'
DATA = BASE / 'data'

AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']
AGENT_LABELS = {
    'geopolitical': 'Geopolitical', 'macro_demand': 'Macro',
    'monetary': 'Monetary', 'supply_opec': 'Supply/OPEC',
    'technical': 'Technical', 'sentiment': 'Sentiment',
    'cross_market': 'Cross-Mkt',
}
REGIME_BINS = [0, 0.20, 0.35, 0.55, 10.0]
REGIME_LABELS = ['Low', 'Normal', 'Elevated', 'Crisis']
GRAY = ['#1a1a1a', '#404040', '#666666', '#8c8c8c', '#b3b3b3', '#d9d9d9']
HATCHES = ['///', '\\\\\\', '|||', '---', '+++', 'xxx', '...']

# ── Matplotlib academic style ─────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})


def save_fig(fig, name):
    path = DOCS / name
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved: {path.name}')


def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def assign_regime(vol):
    return pd.cut(vol, bins=REGIME_BINS, labels=REGIME_LABELS)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_data():
    print('  Loading debate results...')
    df = pd.read_csv(RESULTS / 'debate_eval_full_20260320_2343.csv',
                     parse_dates=['date'])

    # Exclude days contaminated by 2020-04 negative oil price event
    # Forward contamination: forward 20d window includes the extreme day
    # Backward contamination: backward 20d vol includes the extreme day
    n_before = len(df)
    contaminated = (df['actual_vol'] > 2) | (df['persist_vol'] > 2)
    df = df[~contaminated].reset_index(drop=True)
    print(f'  Excluded {n_before - len(df)} days contaminated by negative oil price '
          f'(n: {n_before} -> {len(df)})')

    df['vol_regime'] = assign_regime(df['wti_vol_20d'])

    # Raw errors (the _error columns store squared errors)
    for method in ['debate', 'single', 'har', 'garch', 'persist']:
        df[f'{method}_raw_error'] = df[f'{method}_vol'] - df['actual_vol']

    # Behavioral dummies
    for a in AGENTS:
        col = f'behavior_{a}'
        df[f'herding_{a}'] = (df[col] == 'herding').astype(int)
        df[f'anchored_{a}'] = (df[col] == 'anchored').astype(int)

    # Daily herding proportion
    herd_cols = [f'herding_{a}' for a in AGENTS]
    df['herding_proportion'] = df[herd_cols].sum(axis=1) / 7
    df['herding_count'] = df[herd_cols].sum(axis=1)

    print('  Loading attribution JSON...')
    with open(RESULTS / 'debate_attribution_full_20260320_2343.json') as f:
        attrib = json.load(f)

    print('  Loading baselines...')
    bl_full_path = RESULTS / 'vol_baselines_full.csv'
    bl_dl_path = RESULTS / 'vol_baselines_dl_rolling.csv'
    bl_full = pd.read_csv(bl_full_path, parse_dates=['date']) if bl_full_path.exists() else pd.DataFrame()
    bl_dl = pd.read_csv(bl_dl_path, parse_dates=['date']) if bl_dl_path.exists() else pd.DataFrame()

    print('  Loading macro data...')
    macro = pd.read_csv(DATA / 'oil_macro_daily.csv', parse_dates=['date'])

    return df, attrib, bl_full, bl_dl, macro


# ══════════════════════════════════════════════════════════════════
# MODULE 1: CASE STUDIES
# ══════════════════════════════════════════════════════════════════

CASE_STUDIES = [
    ('covid', '2020-03-01', '2020-06-30', 'COVID Negative Oil',
     [('2020-03-09', 'Oil crash'), ('2020-04-20', 'WTI negative')]),
    ('ukraine', '2022-02-01', '2022-06-30', 'Russia-Ukraine Conflict',
     [('2022-02-24', 'Invasion'), ('2022-03-08', 'Oil peak')]),
    ('mideast', '2025-01-01', '2025-03-31', '2025 Middle East Escalation',
     [('2025-01-15', 'Iran tensions'), ('2025-02-20', 'Escalation')]),
]


def module1_case_studies(df, attrib):
    for tag, start, end, title, events in CASE_STUDIES:
        print(f'  Case: {title}')
        mask = (df['date'] >= start) & (df['date'] <= end)
        sub = df[mask].copy()
        if len(sub) == 0:
            print(f'    No data for {start}~{end}, skipping')
            continue

        # ── Time series figure ──
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(sub['date'], sub['actual_vol'], 'k-', lw=1.5, label='Actual')
        ax.plot(sub['date'], sub['debate_vol'], '--', color=GRAY[1], lw=1.2, label='Debate')
        ax.plot(sub['date'], sub['single_vol'], ':', color=GRAY[2], lw=1.2, label='Single')
        ax.plot(sub['date'], sub['har_vol'], '-.', color=GRAY[3], lw=1.0, label='HAR')
        ax.plot(sub['date'], sub['persist_vol'], '-', color=GRAY[4], lw=0.8, label='Persistence')
        for edate, elabel in events:
            ed = pd.Timestamp(edate)
            if sub['date'].min() <= ed <= sub['date'].max():
                ax.axvline(ed, color='gray', ls='--', lw=0.7, alpha=0.6)
                ax.text(ed, ax.get_ylim()[1] * 0.95, elabel,
                        fontsize=7, ha='center', va='top', rotation=30)
        ax.set_ylabel('Annualized Volatility')
        ax.set_title(title)
        ax.legend(loc='upper right', framealpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        save_fig(fig, f'fig_case_{tag}.pdf')

        # ── Behavioral distribution ──
        full_behaviors = []
        sub_behaviors = []
        for a in AGENTS:
            col = f'behavior_{a}'
            full_behaviors.extend(df[col].tolist())
            sub_behaviors.extend(sub[col].tolist())
        full_dist = pd.Series(full_behaviors).value_counts(normalize=True)
        sub_dist = pd.Series(sub_behaviors).value_counts(normalize=True)
        print(f'    Behavior dist (window vs full):')
        for b in ['herding', 'anchored', 'independent', 'overconfident']:
            sw = sub_dist.get(b, 0) * 100
            fw = full_dist.get(b, 0) * 100
            print(f'      {b:15s}: {sw:5.1f}% vs {fw:5.1f}%')

        # ── Dominant agent ──
        shap_cols = [f'shapley_{a}' for a in AGENTS]
        sub_shap = sub[shap_cols].abs()
        sub_shap.columns = [AGENT_LABELS[a] for a in AGENTS]
        mean_shap = sub_shap.mean().sort_values(ascending=False)
        print(f'    Top Shapley agents: {mean_shap.index[0]} ({mean_shap.iloc[0]:.6f}), '
              f'{mean_shap.index[1]} ({mean_shap.iloc[1]:.6f})')

        # ── RMSE comparison ──
        models = {'Debate': 'debate_vol', 'Single': 'single_vol',
                  'HAR': 'har_vol', 'GARCH': 'garch_vol', 'Persistence': 'persist_vol'}
        print(f'    RMSE in window:')
        for mname, mcol in models.items():
            rmse = np.sqrt(((sub[mcol] - sub['actual_vol']) ** 2).mean())
            print(f'      {mname:12s}: {rmse:.4f}')

        # ── Narrative table (LaTeX) ──
        # Sample ~10 key dates (weekly)
        step = max(1, len(sub) // 10)
        sample = sub.iloc[::step].head(10)
        lines = [
            '% Auto-generated by deep_analysis.py',
            '\\begin{tabular}{lccccl}',
            '\\toprule',
            'Date & Actual & Debate & Error & Dominant & Behavior \\\\',
            '\\midrule',
        ]
        for _, row in sample.iterrows():
            shap_abs = {a: abs(row[f'shapley_{a}']) for a in AGENTS}
            dom = AGENT_LABELS[max(shap_abs, key=shap_abs.get)]
            err = row['debate_raw_error']
            lines.append(
                f"{row['date'].strftime('%Y-%m-%d')} & "
                f"{row['actual_vol']:.3f} & {row['debate_vol']:.3f} & "
                f"{err:+.4f} & {dom} & {row['what']} \\\\"
            )
        lines += ['\\bottomrule', '\\end{tabular}']
        tex_path = DOCS / f'tab_case_{tag}.tex'
        tex_path.write_text('\n'.join(lines))
        print(f'    Saved: {tex_path.name}')


# ══════════════════════════════════════════════════════════════════
# MODULE 2: REGIME-DEPENDENT BEHAVIORAL REGRESSION
# ══════════════════════════════════════════════════════════════════

def module2_behavioral_regression(df, macro):
    merged = df.merge(macro[['date', 'VIX', 'vix_change', 'YIELD_SPREAD_10Y_2Y']],
                      on='date', how='left')
    merged['lagged_vol'] = merged['actual_vol'].shift(1)
    merged['vol_change'] = merged['actual_vol'].diff()
    merged = merged.dropna(subset=['lagged_vol', 'vol_change', 'VIX'])

    # Regime dummies (Low = reference)
    regime_dummies = pd.get_dummies(merged['vol_regime'], dtype=float)
    for col in ['Normal', 'Elevated', 'Crisis']:
        merged[f'r_{col}'] = regime_dummies[col].values if col in regime_dummies else 0

    X_cols = ['r_Normal', 'r_Elevated', 'r_Crisis', 'lagged_vol', 'VIX', 'vol_change']
    X = sm.add_constant(merged[X_cols].astype(float))

    results_herding = {}
    results_anchored = {}

    # ── (A) Agent-level Logit: herding ──
    print('  Logit regressions (herding):')
    for a in AGENTS:
        y = merged[f'herding_{a}'].astype(float)
        try:
            model = Logit(y, X).fit(method='bfgs', maxiter=1000, disp=0)
            results_herding[a] = model
            crisis_coef = model.params.get('r_Crisis', np.nan)
            crisis_p = model.pvalues.get('r_Crisis', np.nan)
            print(f'    {AGENT_LABELS[a]:12s}: Crisis coef={crisis_coef:+.3f}{sig_stars(crisis_p)} '
                  f'(p={crisis_p:.3f}), N={int(model.nobs)}')
        except Exception as e:
            print(f'    {AGENT_LABELS[a]:12s}: Logit failed ({e})')
            results_herding[a] = None

    # ── (B) Agent-level Logit: anchoring ──
    print('  Logit regressions (anchoring):')
    for a in AGENTS:
        y = merged[f'anchored_{a}'].astype(float)
        try:
            model = Logit(y, X).fit(method='bfgs', maxiter=1000, disp=0)
            results_anchored[a] = model
            crisis_coef = model.params.get('r_Crisis', np.nan)
            crisis_p = model.pvalues.get('r_Crisis', np.nan)
            print(f'    {AGENT_LABELS[a]:12s}: Crisis coef={crisis_coef:+.3f}{sig_stars(crisis_p)} '
                  f'(p={crisis_p:.3f})')
        except Exception as e:
            print(f'    {AGENT_LABELS[a]:12s}: Logit failed ({e})')
            results_anchored[a] = None

    # ── (C) Pooled OLS: herding proportion ──
    y_ols = merged['herding_proportion'].astype(float)
    ols_model = sm.OLS(y_ols, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    print(f'  OLS herding proportion: R²={ols_model.rsquared:.4f}')
    print(f'    Crisis: {ols_model.params["r_Crisis"]:+.4f}{sig_stars(ols_model.pvalues["r_Crisis"])}')

    # ── Coefficient plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Crisis coefs for herding
    agents_ok = [a for a in AGENTS if results_herding[a] is not None]
    coefs_h = [results_herding[a].params.get('r_Crisis', 0) for a in agents_ok]
    cis_h = [1.96 * results_herding[a].bse.get('r_Crisis', 0) for a in agents_ok]
    labels = [AGENT_LABELS[a] for a in agents_ok]
    y_pos = range(len(agents_ok))
    ax1.barh(y_pos, coefs_h, xerr=cis_h, color=GRAY[2], edgecolor='black', lw=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.axvline(0, color='black', lw=0.5)
    ax1.set_xlabel('Crisis Coefficient')
    ax1.set_title('(A) Herding Logit')

    # Panel B: Crisis coefs for anchoring
    agents_ok2 = [a for a in AGENTS if results_anchored[a] is not None]
    coefs_a = [results_anchored[a].params.get('r_Crisis', 0) for a in agents_ok2]
    cis_a = [1.96 * results_anchored[a].bse.get('r_Crisis', 0) for a in agents_ok2]
    labels2 = [AGENT_LABELS[a] for a in agents_ok2]
    y_pos2 = range(len(agents_ok2))
    ax2.barh(y_pos2, coefs_a, xerr=cis_a, color=GRAY[3], edgecolor='black', lw=0.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels2)
    ax2.axvline(0, color='black', lw=0.5)
    ax2.set_xlabel('Crisis Coefficient')
    ax2.set_title('(B) Anchoring Logit')

    fig.tight_layout()
    save_fig(fig, 'fig_regime_regression.pdf')

    # ── LaTeX table ──
    lines = [
        '% Auto-generated by deep_analysis.py',
        '\\begin{tabular}{l' + 'c' * (len(AGENTS) + 1) + '}',
        '\\toprule',
        ' & ' + ' & '.join(AGENT_LABELS[a] for a in AGENTS) + ' & OLS \\\\',
        '\\midrule',
        '\\multicolumn{' + str(len(AGENTS) + 2) + '}{l}{\\textit{Panel A: Herding}} \\\\',
    ]
    for var in ['const', 'r_Normal', 'r_Elevated', 'r_Crisis', 'lagged_vol', 'VIX', 'vol_change']:
        row_vals = []
        for a in AGENTS:
            m = results_herding[a]
            if m is not None and var in m.params:
                row_vals.append(f'{m.params[var]:.3f}{sig_stars(m.pvalues[var])}')
            else:
                row_vals.append('--')
        # OLS column
        if var in ols_model.params:
            row_vals.append(f'{ols_model.params[var]:.4f}{sig_stars(ols_model.pvalues[var])}')
        else:
            row_vals.append('--')
        var_label = var.replace('r_', '').replace('const', 'Intercept').replace('lagged_vol', 'Lagged Vol').replace('vol_change', 'Vol Change')
        lines.append(f'{var_label} & ' + ' & '.join(row_vals) + ' \\\\')

    # N and pseudo R2
    n_vals = []
    r2_vals = []
    for a in AGENTS:
        m = results_herding[a]
        if m is not None:
            n_vals.append(f'{int(m.nobs)}')
            r2_vals.append(f'{m.prsquared:.3f}')
        else:
            n_vals.append('--')
            r2_vals.append('--')
    n_vals.append(f'{int(ols_model.nobs)}')
    r2_vals.append(f'{ols_model.rsquared:.3f}')
    lines.append('N & ' + ' & '.join(n_vals) + ' \\\\')
    lines.append('Pseudo $R^2$ & ' + ' & '.join(r2_vals) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}']
    tex_path = DOCS / 'tab_regression.tex'
    tex_path.write_text('\n'.join(lines))
    print(f'  Saved: {tex_path.name}')


# ══════════════════════════════════════════════════════════════════
# MODULE 3: INTERVENTION EXPERIMENT FIX
# ══════════════════════════════════════════════════════════════════

def module3_intervention_fix(df):
    cols = ['intv_original', 'intv_shapley', 'intv_myerson', 'intv_random']
    sub = df[cols + ['vol_regime']].dropna()

    # Absolute error differences (avoid ratio problems)
    sub = sub.copy()
    sub['diff_shapley'] = sub['intv_original'] - sub['intv_shapley']
    sub['diff_myerson'] = sub['intv_original'] - sub['intv_myerson']
    sub['diff_random'] = sub['intv_original'] - sub['intv_random']

    print('  Overall (absolute error difference, positive = improvement):')
    for col_name, label in [('diff_shapley', 'Shapley'), ('diff_myerson', 'Myerson'), ('diff_random', 'Random')]:
        vals = sub[col_name]
        tm = stats.trim_mean(vals, 0.05)
        med = vals.median()
        print(f'    {label:10s}: trimmed_mean={tm:.6f}, median={med:.6f}')

    # Wilcoxon: Shapley vs Random
    stat_sr, p_sr = wilcoxon(sub['intv_shapley'], sub['intv_random'], alternative='less')
    print(f'  Wilcoxon Shapley < Random: stat={stat_sr:.0f}, p={p_sr:.4f}{sig_stars(p_sr)}')

    # Wilcoxon: Shapley vs Original
    stat_so, p_so = wilcoxon(sub['intv_shapley'], sub['intv_original'], alternative='less')
    print(f'  Wilcoxon Shapley < Original: stat={stat_so:.0f}, p={p_so:.4f}{sig_stars(p_so)}')

    # Bootstrap 95% CI for trimmed mean improvement
    rng = np.random.default_rng(42)
    boot_means = []
    vals_arr = sub['diff_shapley'].values
    for _ in range(5000):
        samp = rng.choice(vals_arr, size=len(vals_arr), replace=True)
        boot_means.append(stats.trim_mean(samp, 0.05))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f'  Shapley improvement bootstrap 95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]')

    # ── By regime ──
    print('  By regime:')
    regime_stats = []
    for regime in REGIME_LABELS:
        rsub = sub[sub['vol_regime'] == regime]
        if len(rsub) < 5:
            continue
        orig_mse = rsub['intv_original'].mean()
        shap_mse = rsub['intv_shapley'].mean()
        rand_mse = rsub['intv_random'].mean()
        med_diff = rsub['diff_shapley'].median()
        n = len(rsub)
        # Wilcoxon within regime
        try:
            _, p_reg = wilcoxon(rsub['intv_shapley'], rsub['intv_random'], alternative='less')
        except ValueError:
            p_reg = np.nan
        regime_stats.append((regime, n, orig_mse, shap_mse, rand_mse, med_diff, p_reg))
        print(f'    {regime:10s} (n={n:4d}): median_improvement={med_diff:.6f}, '
              f'Shapley_vs_Random p={p_reg:.4f}')

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Box plots (winsorized for display)
    bp_data = []
    bp_labels = ['Original', 'Shapley', 'Myerson', 'Random']
    for c in ['intv_original', 'intv_shapley', 'intv_myerson', 'intv_random']:
        vals = sub[c].values
        p99 = np.percentile(vals, 99)
        bp_data.append(vals[vals <= p99])
    bp = ax1.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                     medianprops={'color': 'black', 'lw': 1.5})
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(GRAY[i + 1])
    ax1.set_ylabel('Squared Forecast Error')
    ax1.set_title('(A) Intervention Error Distribution')

    # Panel B: By-regime median improvement
    regimes_plot = [r[0] for r in regime_stats]
    med_impr = [r[5] for r in regime_stats]
    colors = [GRAY[i + 1] for i in range(len(regimes_plot))]
    ax2.bar(regimes_plot, med_impr, color=colors, edgecolor='black', lw=0.5)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set_ylabel('Median Error Reduction')
    ax2.set_title('(B) Shapley Intervention by Regime')

    fig.tight_layout()
    save_fig(fig, 'fig_intervention.pdf')

    # ── LaTeX table ──
    lines = [
        '% Auto-generated by deep_analysis.py',
        '\\begin{tabular}{lccccccc}',
        '\\toprule',
        'Regime & N & Original & Shapley & Myerson & Random & Median Impr. & $p$-value \\\\',
        '\\midrule',
    ]
    for regime, n, orig, shap, rand_mse, med_d, p_reg in regime_stats:
        p_str = f'{p_reg:.3f}{sig_stars(p_reg)}' if not np.isnan(p_reg) else '--'
        lines.append(f'{regime} & {n} & {orig:.6f} & {shap:.6f} & -- & {rand_mse:.6f} & '
                     f'{med_d:.6f} & {p_str} \\\\')
    # Overall
    orig_all = sub['intv_original'].mean()
    shap_all = sub['intv_shapley'].mean()
    rand_all = sub['intv_random'].mean()
    med_all = sub['diff_shapley'].median()
    lines.append(f'\\midrule')
    lines.append(f'Overall & {len(sub)} & {orig_all:.6f} & {shap_all:.6f} & -- & {rand_all:.6f} & '
                 f'{med_all:.6f} & {p_sr:.3f}{sig_stars(p_sr)} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (DOCS / 'tab_intervention.tex').write_text('\n'.join(lines))
    print(f'  Saved: tab_intervention.tex')


# ══════════════════════════════════════════════════════════════════
# MODULE 4: SHAPLEY ATTRIBUTION SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════

def module4_shapley_significance(df, attrib):
    shap_cols = [f'shapley_{a}' for a in AGENTS]
    shap_df = df[shap_cols + ['vol_regime']].dropna().copy()
    regime_col = shap_df['vol_regime']
    shap_df = shap_df[shap_cols]
    shap_df.columns = AGENTS

    # Total Shapley each day and equal share
    shap_df['total'] = shap_df[AGENTS].sum(axis=1)
    shap_df['vol_regime'] = regime_col.values
    equal_share = shap_df['total'] / 7

    print('  Bootstrap 95% CI and equal-share test:')
    rng = np.random.default_rng(42)
    agent_results = []

    for a in AGENTS:
        vals = shap_df[a].values
        mean_shap = vals.mean()

        # Bootstrap CI
        boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        # Test vs equal share
        diff = vals - equal_share.values
        t_stat, p_val = ttest_1samp(diff, 0)

        # KW test across regimes
        groups = [shap_df.loc[shap_df['vol_regime'] == r, a].values for r in REGIME_LABELS
                  if (shap_df['vol_regime'] == r).sum() > 0]
        kw_stat, kw_p = kruskal(*groups)

        agent_results.append((a, mean_shap, ci_lo, ci_hi, p_val, kw_stat, kw_p))
        print(f'    {AGENT_LABELS[a]:12s}: mean={mean_shap:+.6f}, '
              f'CI=[{ci_lo:.6f},{ci_hi:.6f}], eq_test p={p_val:.3f}{sig_stars(p_val)}, '
              f'KW p={kw_p:.3f}{sig_stars(kw_p)}')

    # ── Heatmap: agents x regimes ──
    heat = np.zeros((len(AGENTS), len(REGIME_LABELS)))
    for i, a in enumerate(AGENTS):
        for j, r in enumerate(REGIME_LABELS):
            mask = shap_df['vol_regime'] == r
            if mask.sum() > 0:
                heat[i, j] = shap_df.loc[mask, a].abs().mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(heat, cmap='Greys', aspect='auto')
    ax.set_xticks(range(len(REGIME_LABELS)))
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([AGENT_LABELS[a] for a in AGENTS])
    for i in range(len(AGENTS)):
        for j in range(len(REGIME_LABELS)):
            color = 'white' if heat[i, j] > heat.max() * 0.6 else 'black'
            ax.text(j, i, f'{heat[i,j]:.5f}', ha='center', va='center',
                    fontsize=7, color=color)
    ax.set_title('Mean |Shapley| by Agent and Regime')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, 'fig_shapley_heatmap.pdf')

    # ── LaTeX table ──
    lines = [
        '% Auto-generated by deep_analysis.py',
        '\\begin{tabular}{lcccccc}',
        '\\toprule',
        'Agent & Mean Shapley & 95\\% CI & Equal Share $p$ & KW Stat & KW $p$ \\\\',
        '\\midrule',
    ]
    for a, ms, cl, ch, pv, ks, kp in agent_results:
        lines.append(f'{AGENT_LABELS[a]} & {ms:.6f} & [{cl:.6f}, {ch:.6f}] & '
                     f'{pv:.3f}{sig_stars(pv)} & {ks:.2f} & {kp:.3f}{sig_stars(kp)} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (DOCS / 'tab_shapley.tex').write_text('\n'.join(lines))
    print(f'  Saved: tab_shapley.tex')


# ══════════════════════════════════════════════════════════════════
# MODULE 5: INFORMATION CASCADE DYNAMICS
# ══════════════════════════════════════════════════════════════════

def module5_cascade_dynamics(df, attrib):
    df = df.dropna(subset=['debate_raw_error']).copy()
    df['abs_debate_error'] = df['debate_raw_error'].abs()
    df['lagged_vol'] = df['actual_vol'].shift(1)

    # Shapley dispersion (std across agents)
    shap_cols = [f'shapley_{a}' for a in AGENTS]
    df['shapley_std'] = df[shap_cols].std(axis=1)

    # Correlation: herding_proportion vs |error|
    corr_hp, p_hp = stats.spearmanr(df['herding_proportion'], df['abs_debate_error'])
    print(f'  Spearman corr(herding_prop, |error|): r={corr_hp:.3f}, p={p_hp:.4f}{sig_stars(p_hp)}')

    # Correlation: n_influence_edges vs |error|
    corr_ie, p_ie = stats.spearmanr(df['n_influence_edges'], df['abs_debate_error'])
    print(f'  Spearman corr(influence_edges, |error|): r={corr_ie:.3f}, p={p_ie:.4f}{sig_stars(p_ie)}')

    # ── OLS: |error| ~ herding_proportion + regime + lagged_vol ──
    reg_df = df.dropna(subset=['lagged_vol']).copy()
    regime_dummies = pd.get_dummies(reg_df['vol_regime'], dtype=float)
    for col in ['Normal', 'Elevated', 'Crisis']:
        reg_df[f'r_{col}'] = regime_dummies[col].values if col in regime_dummies else 0

    X_vars = ['herding_proportion', 'n_influence_edges', 'r_Normal', 'r_Elevated', 'r_Crisis', 'lagged_vol']
    X = sm.add_constant(reg_df[X_vars].astype(float))
    y = reg_df['abs_debate_error'].astype(float)
    ols = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    print(f'  OLS |error| regression:')
    print(f'    herding_prop: {ols.params["herding_proportion"]:.6f}{sig_stars(ols.pvalues["herding_proportion"])} '
          f'(p={ols.pvalues["herding_proportion"]:.4f})')
    print(f'    influence_edges: {ols.params["n_influence_edges"]:.6f}{sig_stars(ols.pvalues["n_influence_edges"])} '
          f'(p={ols.pvalues["n_influence_edges"]:.4f})')
    print(f'    R²={ols.rsquared:.4f}')

    # ── Figure (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Panel A: Scatter herding_proportion vs |error|
    ax = axes[0]
    regime_markers = {'Low': 'o', 'Normal': 's', 'Elevated': '^', 'Crisis': 'D'}
    regime_grays = {'Low': GRAY[4], 'Normal': GRAY[3], 'Elevated': GRAY[2], 'Crisis': GRAY[0]}
    for regime in REGIME_LABELS:
        mask = df['vol_regime'] == regime
        ax.scatter(df.loc[mask, 'herding_proportion'], df.loc[mask, 'abs_debate_error'],
                   marker=regime_markers[regime], c=regime_grays[regime], s=8, alpha=0.5,
                   label=regime, edgecolors='none')
    # Trend line
    z = np.polyfit(df['herding_proportion'].values, df['abs_debate_error'].values, 1)
    x_line = np.linspace(0, 1, 50)
    ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1)
    ax.set_xlabel('Herding Proportion')
    ax.set_ylabel('|Debate Error|')
    ax.set_title('(A) Herding vs Error')
    ax.legend(fontsize=6, markerscale=1.5)

    # Panel B: Rolling herding + rolling RMSE
    ax = axes[1]
    window = 60
    rolling_hp = df['herding_proportion'].rolling(window).mean()
    rolling_rmse = (df['debate_raw_error'] ** 2).rolling(window).mean().apply(np.sqrt)
    ax.plot(df['date'], rolling_hp, 'k-', lw=1, label='Herding Prop (60d)')
    ax.set_ylabel('Herding Proportion')
    ax.set_xlabel('')
    ax2 = ax.twinx()
    ax2.plot(df['date'], rolling_rmse, '--', color=GRAY[2], lw=1, label='RMSE (60d)')
    ax2.set_ylabel('Rolling RMSE', color=GRAY[2])
    ax.set_title('(B) Rolling Herding & RMSE')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper left')

    # Panel C: Mean |error| by herding count
    ax = axes[2]
    df['herd_bin'] = df['herding_count'].clip(upper=3)
    df.loc[df['herding_count'] >= 3, 'herd_bin'] = 3
    grouped = df.groupby('herd_bin')['abs_debate_error'].agg(['mean', 'sem'])
    x_vals = grouped.index.astype(int)
    ax.bar(x_vals, grouped['mean'], yerr=1.96 * grouped['sem'],
           color=[GRAY[i + 1] for i in range(len(x_vals))], edgecolor='black', lw=0.5,
           capsize=3)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['0', '1', '2', '3+'])
    ax.set_xlabel('Number of Herding Agents')
    ax.set_ylabel('Mean |Debate Error|')
    ax.set_title('(C) Herding Count vs Error')

    fig.tight_layout()
    save_fig(fig, 'fig_cascade.pdf')

    # ── LaTeX table ──
    lines = [
        '% Auto-generated by deep_analysis.py',
        '\\begin{tabular}{lcc}',
        '\\toprule',
        'Variable & Coefficient & $p$-value \\\\',
        '\\midrule',
    ]
    for var in ['const', 'herding_proportion', 'n_influence_edges',
                'r_Normal', 'r_Elevated', 'r_Crisis', 'lagged_vol']:
        label = var.replace('r_', '').replace('const', 'Intercept') \
                   .replace('herding_proportion', 'Herding Proportion') \
                   .replace('n_influence_edges', 'Influence Edges') \
                   .replace('lagged_vol', 'Lagged Vol')
        c = ols.params[var]
        p = ols.pvalues[var]
        se = ols.bse[var]
        lines.append(f'{label} & {c:.6f} ({se:.6f}){sig_stars(p)} & {p:.4f} \\\\')
    lines.append(f'\\midrule')
    lines.append(f'N & \\multicolumn{{2}}{{c}}{{{int(ols.nobs)}}} \\\\')
    lines.append(f'$R^2$ & \\multicolumn{{2}}{{c}}{{{ols.rsquared:.4f}}} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (DOCS / 'tab_cascade.tex').write_text('\n'.join(lines))
    print(f'  Saved: tab_cascade.tex')


# ══════════════════════════════════════════════════════════════════
# MODULE 6: MASTER COMPARISON
# ══════════════════════════════════════════════════════════════════

def diebold_mariano(e1, e2, h=20):
    """DM test with HAC variance. h=20 for 20-day overlapping vol window."""
    d = e1 ** 2 - e2 ** 2
    d = d.dropna()
    T = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        if k < T:
            weight = 1 - k / h  # Bartlett kernel
            cov_k = np.cov(d.values[k:], d.values[:-k])[0, 1]
            gamma_sum += weight * cov_k
    var_d = (gamma_0 + 2 * gamma_sum) / T
    if var_d <= 0:
        var_d = gamma_0 / T

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def module6_comparison(df, bl_full, bl_dl):
    # ── Build master comparison ──
    # Models from debate CSV
    models = {}
    for name, col in [('Debate', 'debate_vol'), ('Single-Agent', 'single_vol'),
                      ('HAR', 'har_vol'), ('GARCH', 'garch_vol'),
                      ('Persistence', 'persist_vol')]:
        models[name] = df[['date', 'actual_vol', col]].rename(columns={col: 'pred'})

    # Models from baselines_full/dl: only include if target matches debate data
    # Check target consistency by comparing actual_vol on overlapping dates
    for bl_name, bl_df in [('baselines_full', bl_full), ('baselines_dl', bl_dl)]:
        if len(bl_df) == 0 or 'model' not in bl_df.columns:
            continue
        # Verify target consistency
        sample_bl = bl_df[['date', 'actual_vol']].drop_duplicates('date').head(10)
        sample_deb = df[['date', 'actual_vol']].head(10)
        merged_check = sample_bl.merge(sample_deb, on='date', suffixes=('_bl', '_deb'))
        if len(merged_check) > 0:
            max_diff = (merged_check['actual_vol_bl'] - merged_check['actual_vol_deb']).abs().max()
            if max_diff > 0.01:
                print(f'  WARNING: {bl_name} uses different target (max diff={max_diff:.4f}), skipping')
                continue
        for mname in bl_df['model'].unique():
            msub = bl_df[bl_df['model'] == mname][['date', 'actual_vol', 'pred_vol']]
            msub = msub.rename(columns={'pred_vol': 'pred'})
            models[mname] = msub

    # ── Compute metrics ──
    results = []
    debate_errors = df['debate_raw_error']

    for mname, mdf in models.items():
        mdf = mdf.dropna()
        err = mdf['pred'] - mdf['actual_vol']
        rmse = np.sqrt((err ** 2).mean())
        mae = err.abs().mean()

        # Merge with regime info
        mdf_reg = mdf.merge(df[['date', 'vol_regime']], on='date', how='inner')

        regime_rmse = {}
        for r in REGIME_LABELS:
            rsub = mdf_reg[mdf_reg['vol_regime'] == r]
            if len(rsub) > 0:
                regime_rmse[r] = np.sqrt(((rsub['pred'] - rsub['actual_vol']) ** 2).mean())
            else:
                regime_rmse[r] = np.nan

        # DM test vs Debate
        if mname != 'Debate':
            merged_dm = mdf.merge(df[['date', 'debate_vol', 'actual_vol']],
                                  on='date', how='inner', suffixes=('', '_deb'))
            e_debate = merged_dm['debate_vol'] - merged_dm['actual_vol_deb']
            e_model = merged_dm['pred'] - merged_dm['actual_vol']
            dm_stat, dm_p = diebold_mariano(e_debate, e_model)
        else:
            dm_stat, dm_p = np.nan, np.nan

        results.append({
            'Model': mname, 'RMSE': rmse, 'MAE': mae,
            'RMSE_Low': regime_rmse.get('Low', np.nan),
            'RMSE_Normal': regime_rmse.get('Normal', np.nan),
            'RMSE_Elevated': regime_rmse.get('Elevated', np.nan),
            'RMSE_Crisis': regime_rmse.get('Crisis', np.nan),
            'DM_stat': dm_stat, 'DM_p': dm_p,
            'N': len(mdf),
        })

    res_df = pd.DataFrame(results).sort_values('RMSE')
    print('  Master comparison:')
    for _, row in res_df.iterrows():
        dm_str = f'DM={row["DM_stat"]:+.2f} p={row["DM_p"]:.3f}' if not np.isnan(row['DM_stat']) else ''
        print(f'    {row["Model"]:15s}: RMSE={row["RMSE"]:.4f}  MAE={row["MAE"]:.4f}  '
              f'N={int(row["N"])}  {dm_str}')

    # ── Figure A: Overall RMSE horizontal bar ──
    fig, ax = plt.subplots(figsize=(6, 4))
    res_sorted = res_df.sort_values('RMSE', ascending=True)
    colors = [GRAY[0] if m == 'Debate' else GRAY[3] for m in res_sorted['Model']]
    ax.barh(range(len(res_sorted)), res_sorted['RMSE'], color=colors, edgecolor='black', lw=0.5)
    ax.set_yticks(range(len(res_sorted)))
    ax.set_yticklabels(res_sorted['Model'])
    ax.set_xlabel('RMSE')
    ax.set_title('Overall RMSE Comparison')
    fig.tight_layout()
    save_fig(fig, 'fig_comparison_overall.pdf')

    # ── Figure B: By-regime grouped bar (top 6) ──
    top6 = res_df.nsmallest(6, 'RMSE')
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(REGIME_LABELS))
    width = 0.12
    for i, (_, row) in enumerate(top6.iterrows()):
        vals = [row.get(f'RMSE_{r}', np.nan) for r in REGIME_LABELS]
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, vals, width, label=row['Model'],
                      color=GRAY[i], edgecolor='black', lw=0.3, hatch=HATCHES[i])
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE by Volatility Regime')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, 'fig_comparison_rmse.pdf')

    # ── LaTeX table ──
    lines = [
        '% Auto-generated by deep_analysis.py',
        '\\begin{tabular}{lccccccccc}',
        '\\toprule',
        'Model & N & RMSE & MAE & Low & Normal & Elevated & Crisis & DM stat & DM $p$ \\\\',
        '\\midrule',
    ]
    for _, row in res_df.iterrows():
        dm_s = f'{row["DM_stat"]:.2f}' if not np.isnan(row['DM_stat']) else '--'
        dm_p = f'{row["DM_p"]:.3f}{sig_stars(row["DM_p"])}' if not np.isnan(row['DM_p']) else '--'
        # Bold best RMSE
        rmse_str = f'\\textbf{{{row["RMSE"]:.4f}}}' if row['Model'] == 'Debate' else f'{row["RMSE"]:.4f}'
        lines.append(
            f'{row["Model"]} & {int(row["N"])} & {rmse_str} & {row["MAE"]:.4f} & '
            f'{row["RMSE_Low"]:.4f} & {row["RMSE_Normal"]:.4f} & '
            f'{row["RMSE_Elevated"]:.4f} & {row["RMSE_Crisis"]:.4f} & '
            f'{dm_s} & {dm_p} \\\\'
        )
    lines += ['\\bottomrule', '\\end{tabular}']
    (DOCS / 'tab_comparison.tex').write_text('\n'.join(lines))
    print(f'  Saved: tab_comparison.tex')

    return res_df


# ══════════════════════════════════════════════════════════════════
# SUMMARY MARKDOWN
# ══════════════════════════════════════════════════════════════════

def write_summary(df):
    md = ['# Deep Analysis Results', '', f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', '',
          '## Files Generated', '',
          '### Figures (PDF)', '']
    figs = ['fig_case_covid', 'fig_case_ukraine', 'fig_case_mideast',
            'fig_regime_regression', 'fig_intervention', 'fig_shapley_heatmap',
            'fig_cascade', 'fig_comparison_overall', 'fig_comparison_rmse']
    for f in figs:
        path = DOCS / f'{f}.pdf'
        status = 'OK' if path.exists() else 'MISSING'
        md.append(f'- `{f}.pdf`: {status}')

    md += ['', '### Tables (LaTeX)', '']
    tabs = ['tab_case_covid', 'tab_case_ukraine', 'tab_case_mideast',
            'tab_regression', 'tab_intervention', 'tab_shapley', 'tab_cascade', 'tab_comparison']
    for t in tabs:
        path = DOCS / f'{t}.tex'
        status = 'OK' if path.exists() else 'MISSING'
        md.append(f'- `{t}.tex`: {status}')

    md += ['', '## Key Findings', '',
           'See individual module outputs above for detailed results.']
    (DOCS / 'deep_analysis_results.md').write_text('\n'.join(md))
    print(f'  Saved: deep_analysis_results.md')


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    DOCS.mkdir(exist_ok=True)
    print('Loading data...')
    df, attrib, bl_full, bl_dl, macro = load_data()

    print('\n=== Module 1: Case Studies ===')
    module1_case_studies(df, attrib)

    print('\n=== Module 2: Regime-Dependent Behavioral Regressions ===')
    module2_behavioral_regression(df, macro)

    print('\n=== Module 3: Intervention Experiment Fix ===')
    module3_intervention_fix(df)

    print('\n=== Module 4: Shapley Attribution Significance ===')
    module4_shapley_significance(df, attrib)

    print('\n=== Module 5: Information Cascade Dynamics ===')
    module5_cascade_dynamics(df, attrib)

    print('\n=== Module 6: Master Comparison ===')
    module6_comparison(df, bl_full, bl_dl)

    print('\n=== Summary ===')
    write_summary(df)

    print('\nAll analysis complete.')


if __name__ == '__main__':
    main()
