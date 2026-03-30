"""
Communication Structure, Counterfactual Pruning, and Causal-GAT Correction Analysis
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE = '/Users/mac/computerscience/17Agent可解释预测'
DEBATE_CSV   = f'{BASE}/results/debate_eval_full_20260320_2343.csv'
ATTRIB_JSON  = f'{BASE}/results/debate_attribution_full_20260320_2343.json'
GAT_CSV      = f'{BASE}/results/causal_gat_results.csv'
OUT_MD       = f'{BASE}/docs/communication_structure_analysis.md'

AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']

# ── Load data ──
df_debate = pd.read_csv(DEBATE_CSV)
with open(ATTRIB_JSON) as f:
    attrib = json.load(f)
df_gat = pd.read_csv(GAT_CSV)

# ── Compute n_herding from behavior columns ──
beh_cols = [f'behavior_{a}' for a in AGENTS]
df_debate['n_herding'] = df_debate[beh_cols].apply(
    lambda row: sum(1 for v in row if v == 'herding'), axis=1)

# ── Build vol_regime ──
def assign_regime(pv):
    if pv < 0.20:
        return 'low'
    elif pv < 0.35:
        return 'normal'
    elif pv < 0.55:
        return 'elevated'
    else:
        return 'high'

df_debate['vol_regime'] = df_debate['persist_vol'].apply(assign_regime)

# ── Clean sample ──
clean = df_debate[(df_debate['actual_vol'] <= 2) & (df_debate['persist_vol'] <= 2)].copy()
print(f"Full sample: {len(df_debate)}, Clean sample: {len(clean)}")
print(f"debate_error is squared error: (pred - actual)^2")
print(f"RMSE = sqrt(mean(debate_error))")
print()

# Also add graph degree sum from JSON
degree_map = {}
for item in attrib:
    degree_map[item['date']] = sum(item['graph_degree'].values())
df_debate['total_degree'] = df_debate['date'].map(degree_map)
clean = df_debate[(df_debate['actual_vol'] <= 2) & (df_debate['persist_vol'] <= 2)].copy()


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 1: Communication Structure Density vs Prediction Quality
# ═══════════════════════════════════════════════════════════════════
print("=" * 72)
print("ANALYSIS 1: Communication Structure Density vs Prediction Quality")
print("=" * 72)

# 1a. Correlation: n_influence_edges vs debate_error
corr_edges_err, p_edges_err = stats.pearsonr(
    clean['n_influence_edges'], clean['debate_error'])
spear_edges_err, sp_p = stats.spearmanr(
    clean['n_influence_edges'], clean['debate_error'])
print(f"\n1a. n_influence_edges vs debate_error (squared error):")
print(f"    Pearson  r = {corr_edges_err:.4f}, p = {p_edges_err:.4e}")
print(f"    Spearman r = {spear_edges_err:.4f}, p = {sp_p:.4e}")

# Also with RMSE-like metric: absolute error = sqrt(debate_error)
clean['abs_error'] = np.sqrt(clean['debate_error'])
corr_abs, p_abs = stats.pearsonr(clean['n_influence_edges'], clean['abs_error'])
print(f"    n_edges vs |error|: Pearson r = {corr_abs:.4f}, p = {p_abs:.4e}")

# 1b. Quartile analysis
clean['edge_quartile'] = pd.qcut(clean['n_influence_edges'], 4,
                                  labels=['Q1 (sparse)', 'Q2', 'Q3', 'Q4 (dense)'],
                                  duplicates='drop')
quartile_stats = clean.groupby('edge_quartile', observed=False).agg(
    n=('debate_error', 'count'),
    mean_edges=('n_influence_edges', 'mean'),
    RMSE=('debate_error', lambda x: np.sqrt(x.mean())),
    mean_abs_err=('abs_error', 'mean'),
    mean_herding=('n_herding', 'mean'),
    mean_persist_vol=('persist_vol', 'mean'),
    mean_actual_vol=('actual_vol', 'mean'),
).reset_index()

print(f"\n1b. RMSE by n_influence_edges quartile:")
print(quartile_stats.to_string(index=False, float_format='%.4f'))

# 1c. Correlation: n_influence_edges vs n_herding
corr_edges_herd, p_eh = stats.pearsonr(
    clean['n_influence_edges'], clean['n_herding'])
print(f"\n1c. n_influence_edges vs n_herding:")
print(f"    Pearson r = {corr_edges_herd:.4f}, p = {p_eh:.4e}")

# Cross-tab: mean n_herding by edge quartile (already shown above)
# 1d. Does denser communication lead to worse predictions?
# Compare Q1 vs Q4 RMSE
q1_rmse = np.sqrt(clean[clean['edge_quartile'] == 'Q1 (sparse)']['debate_error'].mean())
q4_rmse = np.sqrt(clean[clean['edge_quartile'] == 'Q4 (dense)']['debate_error'].mean())
print(f"\n1d. Q1 (sparse) RMSE = {q1_rmse:.4f}, Q4 (dense) RMSE = {q4_rmse:.4f}")
print(f"    Difference (Q4 - Q1) = {q4_rmse - q1_rmse:.4f}")
# t-test on squared errors
q1_errs = clean[clean['edge_quartile'] == 'Q1 (sparse)']['debate_error']
q4_errs = clean[clean['edge_quartile'] == 'Q4 (dense)']['debate_error']
t_stat, t_p = stats.mannwhitneyu(q1_errs, q4_errs, alternative='two-sided')
print(f"    Mann-Whitney U test: U = {t_stat:.1f}, p = {t_p:.4e}")

# 1e. Regression: debate_error ~ n_influence_edges + n_herding + vol_regime
print(f"\n1e. OLS Regression: debate_error ~ n_influence_edges + n_herding + vol_regime")
reg_df = clean[['debate_error', 'n_influence_edges', 'n_herding', 'vol_regime']].dropna()
regime_dummies = pd.get_dummies(reg_df['vol_regime'], prefix='regime', drop_first=True, dtype=float)
X = pd.concat([reg_df[['n_influence_edges', 'n_herding']].astype(float), regime_dummies], axis=1)
X = sm.add_constant(X)
y = reg_df['debate_error']
model = sm.OLS(y, X).fit(cov_type='HC1')
print(model.summary2().tables[1].to_string(float_format='%.6f'))
print(f"    R-squared = {model.rsquared:.4f}, Adj R-squared = {model.rsquared_adj:.4f}")
print(f"    F-stat = {model.fvalue:.2f}, p = {model.f_pvalue:.4e}")

# Also do regression with log(debate_error) to handle skewness
print(f"\n    Robustness: log(debate_error) regression")
y_log = np.log(reg_df['debate_error'] + 1e-10)
model_log = sm.OLS(y_log, X).fit(cov_type='HC1')
print(model_log.summary2().tables[1].to_string(float_format='%.6f'))
print(f"    R-squared = {model_log.rsquared:.4f}")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Counterfactual Causal Pruning
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ANALYSIS 2: Counterfactual Causal Pruning")
print("=" * 72)

intv_cols = ['intv_original', 'intv_shapley', 'intv_myerson', 'intv_random',
             'intv_shapley_reduction', 'intv_myerson_reduction', 'intv_random_reduction']
intv = clean.dropna(subset=['intv_original']).copy()

# 2a. Average error reduction
print(f"\n2a. Average intervention error reduction (proportional):")
print(f"    Shapley reduction: mean = {intv['intv_shapley_reduction'].mean():.4f}, "
      f"median = {intv['intv_shapley_reduction'].median():.4f}")
print(f"    Myerson reduction: mean = {intv['intv_myerson_reduction'].mean():.4f}, "
      f"median = {intv['intv_myerson_reduction'].median():.4f}")
print(f"    Random  reduction: mean = {intv['intv_random_reduction'].mean():.4f}, "
      f"median = {intv['intv_random_reduction'].median():.4f}")

# Positive reduction = intervention helped; negative = hurt
print(f"\n    Fraction of days where intervention helped (reduction > 0):")
print(f"    Shapley: {(intv['intv_shapley_reduction'] > 0).mean():.3f}")
print(f"    Myerson: {(intv['intv_myerson_reduction'] > 0).mean():.3f}")
print(f"    Random:  {(intv['intv_random_reduction'] > 0).mean():.3f}")

# RMSE comparison
rmse_orig = np.sqrt(intv['intv_original'].mean())
rmse_shap = np.sqrt(intv['intv_shapley'].mean())
rmse_myer = np.sqrt(intv['intv_myerson'].mean())
rmse_rand = np.sqrt(intv['intv_random'].mean())
print(f"\n    RMSE comparison:")
print(f"    Original debate:     {rmse_orig:.4f}")
print(f"    Shapley-intervened:  {rmse_shap:.4f} ({(rmse_shap/rmse_orig - 1)*100:+.2f}%)")
print(f"    Myerson-intervened:  {rmse_myer:.4f} ({(rmse_myer/rmse_orig - 1)*100:+.2f}%)")
print(f"    Random-intervened:   {rmse_rand:.4f} ({(rmse_rand/rmse_orig - 1)*100:+.2f}%)")

# 2b. Intervention effectiveness by n_herding
print(f"\n2b. Intervention effectiveness by n_herding:")
herd_intv = intv.groupby('n_herding').agg(
    n=('intv_original', 'count'),
    orig_RMSE=('intv_original', lambda x: np.sqrt(x.mean())),
    shap_RMSE=('intv_shapley', lambda x: np.sqrt(x.mean())),
    myer_RMSE=('intv_myerson', lambda x: np.sqrt(x.mean())),
    shap_reduction=('intv_shapley_reduction', 'mean'),
    myer_reduction=('intv_myerson_reduction', 'mean'),
    rand_reduction=('intv_random_reduction', 'mean'),
).reset_index()
print(herd_intv.to_string(index=False, float_format='%.4f'))

# 2c. Intervention effectiveness by regime
print(f"\n2c. Intervention effectiveness by vol_regime:")
regime_intv = intv.groupby('vol_regime').agg(
    n=('intv_original', 'count'),
    orig_RMSE=('intv_original', lambda x: np.sqrt(x.mean())),
    shap_RMSE=('intv_shapley', lambda x: np.sqrt(x.mean())),
    shap_reduction=('intv_shapley_reduction', 'mean'),
    myer_reduction=('intv_myerson_reduction', 'mean'),
).reset_index()
# Order regimes
regime_order = ['low', 'normal', 'elevated', 'high']
regime_intv['vol_regime'] = pd.Categorical(regime_intv['vol_regime'],
                                            categories=regime_order, ordered=True)
regime_intv = regime_intv.sort_values('vol_regime')
print(regime_intv.to_string(index=False, float_format='%.4f'))

# 2d. Counterfactual: debate vs single when herding is high
print(f"\n2d. Debate vs Single-agent by herding level:")
print(f"    When herding is high, does single-agent outperform debate?")
for nh in sorted(intv['n_herding'].unique()):
    sub = intv[intv['n_herding'] == nh]
    if len(sub) < 10:
        continue
    rmse_deb = np.sqrt(sub['debate_error'].mean())
    rmse_sin = np.sqrt(sub['single_error'].mean())
    rmse_per = np.sqrt(sub['persist_error'].mean())
    winner = 'debate' if rmse_deb < rmse_sin else 'single'
    print(f"    n_herding={nh}: n={len(sub)}, "
          f"debate_RMSE={rmse_deb:.4f}, single_RMSE={rmse_sin:.4f}, "
          f"persist_RMSE={rmse_per:.4f} -> {winner} wins")

# 2e. Deherded counterfactual
print(f"\n2e. 'Deherded' counterfactual approximation:")
print(f"    Using single_vol when n_herding >= threshold vs debate_vol always:")
for threshold in [3, 4, 5]:
    sub = intv.copy()
    sub['deherded_vol'] = np.where(
        sub['n_herding'] >= threshold,
        sub['single_vol'],
        sub['debate_vol']
    )
    sub['deherded_error'] = (sub['deherded_vol'] - sub['actual_vol'])**2
    rmse_deh = np.sqrt(sub['deherded_error'].mean())
    rmse_deb = np.sqrt(sub['debate_error'].mean())
    n_switched = (sub['n_herding'] >= threshold).sum()
    print(f"    Switch to single when n_herding >= {threshold}: "
          f"n_switched={n_switched}, deherded_RMSE={rmse_deh:.4f} vs debate_RMSE={rmse_deb:.4f} "
          f"({(rmse_deh/rmse_deb - 1)*100:+.2f}%)")


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Causal-GAT Correction Case Studies
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ANALYSIS 3: Causal-GAT Correction Case Studies")
print("=" * 72)

gat = df_gat.copy()
# Clean sample for GAT
gat = gat[(gat['actual_vol'] <= 2) & (gat['persist_vol'] <= 2)].copy()
print(f"\nGAT clean sample: n = {len(gat)}")

gat['gat_error'] = (gat['gat_pred'] - gat['actual_vol'])**2
gat['gat_correction'] = np.abs(gat['gat_pred'] - gat['debate_vol'])
gat['abs_error_debate'] = np.sqrt(gat['debate_error'])
gat['abs_error_gat'] = np.sqrt(gat['gat_error'])

# Overall GAT performance
rmse_gat = np.sqrt(gat['gat_error'].mean())
rmse_deb = np.sqrt(gat['debate_error'].mean())
rmse_har = np.sqrt(gat['har_error'].mean())
rmse_per = np.sqrt(gat['persist_error'].mean())
print(f"\nOverall RMSE (GAT sample, n={len(gat)}):")
print(f"  GAT:      {rmse_gat:.4f}")
print(f"  Debate:   {rmse_deb:.4f}")
print(f"  HAR:      {rmse_har:.4f}")
print(f"  Persist:  {rmse_per:.4f}")

# 3a. Top 20 days with largest GAT correction
top20 = gat.nlargest(20, 'gat_correction').copy()
top20['gat_helped'] = top20['gat_error'] < top20['debate_error']

print(f"\n3a. Top 20 days with largest |GAT correction| (|gat_pred - debate_vol|):")
print(f"{'date':>12s} {'actual':>8s} {'debate':>8s} {'gat_pred':>8s} {'har':>8s} {'persist':>8s} "
      f"{'deb_err':>8s} {'gat_err':>8s} {'helped':>7s} {'n_herd':>7s} "
      f"{'w_deb':>6s} {'w_har':>6s} {'w_per':>6s} {'w_sin':>6s} {'w_gar':>6s}")
print("-" * 130)
for _, row in top20.iterrows():
    print(f"{row['date']:>12s} {row['actual_vol']:8.4f} {row['debate_vol']:8.4f} "
          f"{row['gat_pred']:8.4f} {row['har_vol']:8.4f} {row['persist_vol']:8.4f} "
          f"{np.sqrt(row['debate_error']):8.4f} {np.sqrt(row['gat_error']):8.4f} "
          f"{'YES' if row['gat_helped'] else 'no':>7s} "
          f"{int(row['n_herding']):>7d} "
          f"{row['w_debate']:6.3f} {row['w_har']:6.3f} {row['w_persist']:6.3f} "
          f"{row['w_single']:6.3f} {row['w_garch']:6.3f}")

# Summary stats
n_helped = top20['gat_helped'].sum()
print(f"\n    GAT correction helped: {n_helped}/20 ({n_helped/20*100:.0f}%)")
print(f"    Mean n_herding in top 20: {top20['n_herding'].mean():.2f} "
      f"(overall mean: {gat['n_herding'].mean():.2f})")

# 3b. Categorize by correction direction and outcome
gat['correction_direction'] = np.where(
    gat['gat_pred'] > gat['debate_vol'], 'upward', 'downward')
gat['correction_helped'] = gat['gat_error'] < gat['debate_error']

print(f"\n3b. GAT correction direction and outcome:")
ct = gat.groupby(['correction_direction', 'correction_helped']).size().unstack(fill_value=0)
print(ct)
print(f"\n    Overall: GAT better than debate on {gat['correction_helped'].sum()}/{len(gat)} days "
      f"({gat['correction_helped'].mean()*100:.1f}%)")

# 3c. GAT performance by herding level
print(f"\n3c. GAT performance by n_herding:")
gat_herd = gat.groupby('n_herding').agg(
    n=('gat_error', 'count'),
    gat_RMSE=('gat_error', lambda x: np.sqrt(x.mean())),
    debate_RMSE=('debate_error', lambda x: np.sqrt(x.mean())),
    har_RMSE=('har_error', lambda x: np.sqrt(x.mean())),
    gat_better_pct=('correction_helped', 'mean'),
    mean_w_debate=('w_debate', 'mean'),
    mean_w_har=('w_har', 'mean'),
    mean_w_persist=('w_persist', 'mean'),
).reset_index()
print(gat_herd.to_string(index=False, float_format='%.4f'))

# 3d. GAT weights analysis
print(f"\n3d. GAT weight statistics:")
w_cols = ['w_debate', 'w_har', 'w_persist', 'w_single', 'w_garch']
for wc in w_cols:
    print(f"    {wc}: mean={gat[wc].mean():.4f}, median={gat[wc].median():.4f}, "
          f"std={gat[wc].std():.4f}, min={gat[wc].min():.4f}, max={gat[wc].max():.4f}")

# How GAT weights shift with herding
print(f"\n    GAT weight shift by herding level:")
gat_w_herd = gat.groupby('n_herding')[w_cols].mean()
print(gat_w_herd.to_string(float_format='%.4f'))

# 3e. GAT performance by regime
print(f"\n3e. GAT performance by vol_regime:")
gat['vol_regime'] = gat['persist_vol'].apply(assign_regime)
gat_regime = gat.groupby('vol_regime').agg(
    n=('gat_error', 'count'),
    gat_RMSE=('gat_error', lambda x: np.sqrt(x.mean())),
    debate_RMSE=('debate_error', lambda x: np.sqrt(x.mean())),
    gat_better_pct=('correction_helped', 'mean'),
).reset_index()
gat_regime['vol_regime'] = pd.Categorical(gat_regime['vol_regime'],
                                           categories=regime_order, ordered=True)
gat_regime = gat_regime.sort_values('vol_regime')
print(gat_regime.to_string(index=False, float_format='%.4f'))


# ═══════════════════════════════════════════════════════════════════
# SAVE MARKDOWN SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("Saving summary to docs/communication_structure_analysis.md")
print("=" * 72)

md_lines = []
md_lines.append("# Communication Structure, Counterfactual Pruning, and GAT Correction Analysis\n")
md_lines.append(f"Clean sample: n = {len(clean)} (excluding actual_vol > 2 or persist_vol > 2)\n")

# Analysis 1
md_lines.append("## 1. Communication Structure Density vs Prediction Quality\n")
md_lines.append("### 1a. Correlation between influence edges and prediction error\n")
md_lines.append(f"| Metric | r | p-value |")
md_lines.append(f"|--------|---|---------|")
md_lines.append(f"| Pearson (edges vs SE) | {corr_edges_err:.4f} | {p_edges_err:.4e} |")
md_lines.append(f"| Spearman (edges vs SE) | {spear_edges_err:.4f} | {sp_p:.4e} |")
md_lines.append(f"| Pearson (edges vs AE) | {corr_abs:.4f} | {p_abs:.4e} |")
md_lines.append(f"| Pearson (edges vs n_herding) | {corr_edges_herd:.4f} | {p_eh:.4e} |")
md_lines.append("")

md_lines.append("### 1b. RMSE by influence edge density quartile\n")
md_lines.append("| Quartile | n | Mean Edges | RMSE | Mean Herding | Mean Persist Vol |")
md_lines.append("|----------|---|-----------|------|--------------|-----------------|")
for _, row in quartile_stats.iterrows():
    md_lines.append(f"| {row['edge_quartile']} | {int(row['n'])} | {row['mean_edges']:.1f} | "
                    f"{row['RMSE']:.4f} | {row['mean_herding']:.2f} | {row['mean_persist_vol']:.4f} |")
md_lines.append("")

md_lines.append("### 1d. Sparse vs Dense comparison\n")
md_lines.append(f"- Q1 (sparse) RMSE: {q1_rmse:.4f}")
md_lines.append(f"- Q4 (dense) RMSE: {q4_rmse:.4f}")
md_lines.append(f"- Difference (Q4 - Q1): {q4_rmse - q1_rmse:.4f}")
md_lines.append(f"- Mann-Whitney U p-value: {t_p:.4e}")
md_lines.append("")

md_lines.append("### 1e. OLS Regression: debate_error ~ edges + herding + regime\n")
md_lines.append("```")
md_lines.append(model.summary2().tables[1].to_string(float_format='%.6f'))
md_lines.append(f"R-squared = {model.rsquared:.4f}, Adj R-squared = {model.rsquared_adj:.4f}")
md_lines.append("```\n")

md_lines.append("Robustness check with log(debate_error):\n")
md_lines.append("```")
md_lines.append(model_log.summary2().tables[1].to_string(float_format='%.6f'))
md_lines.append(f"R-squared = {model_log.rsquared:.4f}")
md_lines.append("```\n")

# Analysis 2
md_lines.append("## 2. Counterfactual Causal Pruning\n")
md_lines.append("### 2a. Average intervention error reduction\n")
md_lines.append("| Method | Mean Reduction | Median Reduction | Fraction Helped | RMSE |")
md_lines.append("|--------|---------------|-----------------|-----------------|------|")
md_lines.append(f"| Original | - | - | - | {rmse_orig:.4f} |")
md_lines.append(f"| Shapley | {intv['intv_shapley_reduction'].mean():.4f} | "
                f"{intv['intv_shapley_reduction'].median():.4f} | "
                f"{(intv['intv_shapley_reduction'] > 0).mean():.3f} | {rmse_shap:.4f} |")
md_lines.append(f"| Myerson | {intv['intv_myerson_reduction'].mean():.4f} | "
                f"{intv['intv_myerson_reduction'].median():.4f} | "
                f"{(intv['intv_myerson_reduction'] > 0).mean():.3f} | {rmse_myer:.4f} |")
md_lines.append(f"| Random | {intv['intv_random_reduction'].mean():.4f} | "
                f"{intv['intv_random_reduction'].median():.4f} | "
                f"{(intv['intv_random_reduction'] > 0).mean():.3f} | {rmse_rand:.4f} |")
md_lines.append("")

md_lines.append("### 2b. Intervention effectiveness by n_herding\n")
md_lines.append("| n_herding | n | Orig RMSE | Shapley RMSE | Myerson RMSE | Shapley Reduct | Myerson Reduct |")
md_lines.append("|-----------|---|-----------|-------------|-------------|---------------|---------------|")
for _, row in herd_intv.iterrows():
    md_lines.append(f"| {int(row['n_herding'])} | {int(row['n'])} | {row['orig_RMSE']:.4f} | "
                    f"{row['shap_RMSE']:.4f} | {row['myer_RMSE']:.4f} | "
                    f"{row['shap_reduction']:.4f} | {row['myer_reduction']:.4f} |")
md_lines.append("")

md_lines.append("### 2c. Intervention effectiveness by vol_regime\n")
md_lines.append("| Regime | n | Orig RMSE | Shapley RMSE | Shapley Reduct | Myerson Reduct |")
md_lines.append("|--------|---|-----------|-------------|---------------|---------------|")
for _, row in regime_intv.iterrows():
    md_lines.append(f"| {row['vol_regime']} | {int(row['n'])} | {row['orig_RMSE']:.4f} | "
                    f"{row['shap_RMSE']:.4f} | {row['shap_reduction']:.4f} | "
                    f"{row['myer_reduction']:.4f} |")
md_lines.append("")

md_lines.append("### 2d. Debate vs Single-agent by herding level\n")
md_lines.append("| n_herding | n | Debate RMSE | Single RMSE | Winner |")
md_lines.append("|-----------|---|------------|------------|--------|")
for nh in sorted(intv['n_herding'].unique()):
    sub = intv[intv['n_herding'] == nh]
    if len(sub) < 10:
        continue
    rmse_deb_nh = np.sqrt(sub['debate_error'].mean())
    rmse_sin_nh = np.sqrt(sub['single_error'].mean())
    winner = 'debate' if rmse_deb_nh < rmse_sin_nh else 'single'
    md_lines.append(f"| {nh} | {len(sub)} | {rmse_deb_nh:.4f} | {rmse_sin_nh:.4f} | {winner} |")
md_lines.append("")

md_lines.append("### 2e. Deherded counterfactual (switch to single when herding high)\n")
md_lines.append("| Threshold | n Switched | Deherded RMSE | Debate RMSE | Change |")
md_lines.append("|-----------|-----------|--------------|------------|--------|")
for threshold in [3, 4, 5]:
    sub = intv.copy()
    sub['deherded_vol'] = np.where(sub['n_herding'] >= threshold, sub['single_vol'], sub['debate_vol'])
    sub['deherded_error'] = (sub['deherded_vol'] - sub['actual_vol'])**2
    rmse_deh = np.sqrt(sub['deherded_error'].mean())
    rmse_deb_all = np.sqrt(sub['debate_error'].mean())
    n_sw = (sub['n_herding'] >= threshold).sum()
    md_lines.append(f"| >= {threshold} | {n_sw} | {rmse_deh:.4f} | {rmse_deb_all:.4f} | "
                    f"{(rmse_deh/rmse_deb_all - 1)*100:+.2f}% |")
md_lines.append("")

# Analysis 3
md_lines.append("## 3. Causal-GAT Correction Case Studies\n")
md_lines.append(f"GAT clean sample: n = {len(gat)}\n")
md_lines.append("### Overall RMSE comparison (GAT sample)\n")
md_lines.append(f"| Model | RMSE |")
md_lines.append(f"|-------|------|")
md_lines.append(f"| Causal-GAT | {rmse_gat:.4f} |")
rmse_deb_gat = np.sqrt(gat['debate_error'].mean())
md_lines.append(f"| Debate | {rmse_deb_gat:.4f} |")
md_lines.append(f"| HAR | {rmse_har:.4f} |")
md_lines.append(f"| Persistence | {rmse_per:.4f} |")
md_lines.append("")

md_lines.append("### 3a. Top 20 largest GAT corrections\n")
md_lines.append("| Date | Actual | Debate | GAT | HAR | Persist | Debate AE | GAT AE | Helped | n_herd |")
md_lines.append("|------|--------|--------|-----|-----|---------|----------|--------|--------|--------|")
for _, row in top20.iterrows():
    md_lines.append(f"| {row['date']} | {row['actual_vol']:.4f} | {row['debate_vol']:.4f} | "
                    f"{row['gat_pred']:.4f} | {row['har_vol']:.4f} | {row['persist_vol']:.4f} | "
                    f"{np.sqrt(row['debate_error']):.4f} | {np.sqrt(row['gat_error']):.4f} | "
                    f"{'YES' if row['gat_helped'] else 'no'} | {int(row['n_herding'])} |")
md_lines.append("")
md_lines.append(f"GAT correction helped in {n_helped}/20 cases ({n_helped/20*100:.0f}%)\n")
md_lines.append(f"Mean n_herding in top 20: {top20['n_herding'].mean():.2f} (overall: {gat['n_herding'].mean():.2f})\n")

md_lines.append("### 3c. GAT performance by herding level\n")
md_lines.append("| n_herding | n | GAT RMSE | Debate RMSE | HAR RMSE | GAT Better % | w_debate | w_har | w_persist |")
md_lines.append("|-----------|---|----------|------------|---------|-------------|----------|-------|-----------|")
for _, row in gat_herd.iterrows():
    md_lines.append(f"| {int(row['n_herding'])} | {int(row['n'])} | {row['gat_RMSE']:.4f} | "
                    f"{row['debate_RMSE']:.4f} | {row['har_RMSE']:.4f} | "
                    f"{row['gat_better_pct']:.3f} | {row['mean_w_debate']:.4f} | "
                    f"{row['mean_w_har']:.4f} | {row['mean_w_persist']:.4f} |")
md_lines.append("")

md_lines.append("### 3d. GAT weight statistics\n")
md_lines.append("| Weight | Mean | Median | Std | Min | Max |")
md_lines.append("|--------|------|--------|-----|-----|-----|")
for wc in w_cols:
    md_lines.append(f"| {wc} | {gat[wc].mean():.4f} | {gat[wc].median():.4f} | "
                    f"{gat[wc].std():.4f} | {gat[wc].min():.4f} | {gat[wc].max():.4f} |")
md_lines.append("")

md_lines.append("### GAT weight shift by herding level\n")
md_lines.append(gat_w_herd.to_markdown(floatfmt='.4f'))
md_lines.append("")

md_lines.append("### 3e. GAT performance by vol_regime\n")
md_lines.append("| Regime | n | GAT RMSE | Debate RMSE | GAT Better % |")
md_lines.append("|--------|---|----------|------------|-------------|")
for _, row in gat_regime.iterrows():
    md_lines.append(f"| {row['vol_regime']} | {int(row['n'])} | {row['gat_RMSE']:.4f} | "
                    f"{row['debate_RMSE']:.4f} | {row['gat_better_pct']:.3f} |")
md_lines.append("")

with open(OUT_MD, 'w') as f:
    f.write('\n'.join(md_lines))

print(f"\nDone. Summary saved to {OUT_MD}")
