"""
Quantify GDELT signal lead time over WTI oil price regime shifts.
Core empirical evidence for proactive concept drift detection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

# ── Load data ──────────────────────────────────────────────
wti = pd.read_csv('data/wti_full.csv', parse_dates=['date'])
gdelt = pd.read_csv('data/gdelt_daily_features.csv', parse_dates=['date'])

df = wti.merge(gdelt, on='date', how='inner')
df = df.sort_values('date').reset_index(drop=True)
print(f"Merged dataset: {len(df)} days, {df['date'].min().date()} ~ {df['date'].max().date()}")

# ── Compute regime indicators ──────────────────────────────
# Realized volatility (20-day rolling)
df['realized_vol'] = df['wti_return'].rolling(20).std() * np.sqrt(252)

# High-vol regime: top 15% of realized volatility
vol_threshold = df['realized_vol'].quantile(0.85)
df['high_vol_regime'] = (df['realized_vol'] > vol_threshold).astype(int)

# Regime shift indicator: transition from low-vol to high-vol
df['regime_shift'] = (df['high_vol_regime'].diff() == 1).astype(int)

# ── GDELT conflict intensity index ────────────────────────
# Normalize conflict metrics
df['conflict_intensity'] = (
    df['oil_material_conflict_share'] * 0.4 +
    df['oil_conflict_share'] * 0.3 +
    (-df['oil_goldstein_mean'] / 10).clip(0, 1) * 0.3
)

# Smooth with 7-day rolling average
df['conflict_intensity_7d'] = df['conflict_intensity'].rolling(7).mean()
df['conflict_intensity_21d'] = df['conflict_intensity'].rolling(21).mean()

# Z-score for anomaly detection
df['conflict_zscore'] = (
    (df['conflict_intensity_7d'] - df['conflict_intensity_7d'].rolling(60).mean()) /
    df['conflict_intensity_7d'].rolling(60).std()
)

print(f"\nRegime shifts detected: {df['regime_shift'].sum()}")
print(f"High-vol days: {df['high_vol_regime'].sum()} ({df['high_vol_regime'].mean()*100:.1f}%)")

# ── Cross-correlation analysis ─────────────────────────────
print("\n" + "="*60)
print("CROSS-CORRELATION: GDELT conflict → WTI volatility")
print("="*60)

# Compute cross-correlation at different lags
max_lag = 30
valid = df.dropna(subset=['conflict_intensity_7d', 'realized_vol'])
x = valid['conflict_intensity_7d'].values
y = valid['realized_vol'].values

# Standardize
x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

lags = range(-max_lag, max_lag + 1)
ccf = [np.corrcoef(x[max_lag:len(x)-max_lag],
                     np.roll(y, -lag)[max_lag:len(y)-max_lag])[0, 1]
       for lag in lags]

# Positive lag = GDELT leads (what we want to show)
print("\nCross-correlation (positive lag = GDELT leads oil volatility):")
for lag in [0, 5, 10, 15, 20, 25, 30]:
    idx = lag + max_lag
    print(f"  Lag {lag:2d} days: r = {ccf[idx]:.4f}")

peak_lag = lags[np.argmax(ccf)]
peak_corr = max(ccf)
print(f"\nPeak correlation: r = {peak_corr:.4f} at lag = {peak_lag} days")

# ── Granger causality tests ───────────────────────────────
print("\n" + "="*60)
print("GRANGER CAUSALITY: GDELT conflict → WTI volatility")
print("="*60)

gc_data = df[['realized_vol', 'conflict_intensity_7d']].dropna()
for lag in [5, 10, 15, 20]:
    try:
        result = grangercausalitytests(gc_data, maxlag=lag, verbose=False)
        f_stat = result[lag][0]['ssr_ftest'][0]
        p_val = result[lag][0]['ssr_ftest'][1]
        print(f"  Lag {lag:2d}: F = {f_stat:.3f}, p = {p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
    except Exception as e:
        print(f"  Lag {lag:2d}: Error - {e}")

# ── Regime-conditional Granger causality ──────────────────
print("\n" + "="*60)
print("REGIME-CONDITIONAL GRANGER CAUSALITY")
print("="*60)

for regime_name, regime_val in [("Low-Vol", 0), ("High-Vol", 1)]:
    subset = df[df['high_vol_regime'] == regime_val][['realized_vol', 'conflict_intensity_7d']].dropna()
    if len(subset) < 50:
        print(f"  {regime_name}: insufficient data ({len(subset)} obs)")
        continue
    for lag in [5, 10]:
        try:
            result = grangercausalitytests(subset, maxlag=lag, verbose=False)
            f_stat = result[lag][0]['ssr_ftest'][0]
            p_val = result[lag][0]['ssr_ftest'][1]
            print(f"  {regime_name}, Lag {lag:2d}: F = {f_stat:.3f}, p = {p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
        except:
            pass

# ── Event study: major regime shift episodes ──────────────
print("\n" + "="*60)
print("EVENT STUDY: GDELT signals around major regime shifts")
print("="*60)

# Find major regime shift episodes
shift_dates = df[df['regime_shift'] == 1]['date'].tolist()

# Known major events
major_events = {
    '2014 Oil Price Crash': ('2014-06-01', '2014-12-31'),
    '2020 COVID + Price War': ('2020-02-01', '2020-05-31'),
    '2022 Russia-Ukraine': ('2022-01-01', '2022-06-30'),
    '2026 Iran-US Conflict': ('2025-12-01', '2026-03-13'),
}

for event_name, (start, end) in major_events.items():
    mask = (df['date'] >= start) & (df['date'] <= end)
    if mask.sum() == 0:
        print(f"\n{event_name}: No data in range")
        continue

    sub = df[mask].copy()

    # Find first regime shift in this period
    shifts_in_period = sub[sub['regime_shift'] == 1]

    # Find first conflict z-score spike (>2)
    conflict_spikes = sub[sub['conflict_zscore'] > 2]

    # Find first major price move (>3% daily)
    price_moves = sub[sub['wti_return'].abs() > 0.03]

    print(f"\n{event_name} ({start} ~ {end}):")
    print(f"  First conflict z-score > 2: {conflict_spikes['date'].iloc[0].date() if len(conflict_spikes) > 0 else 'None'}")
    print(f"  First regime shift:         {shifts_in_period['date'].iloc[0].date() if len(shifts_in_period) > 0 else 'None'}")
    print(f"  First >3% price move:       {price_moves['date'].iloc[0].date() if len(price_moves) > 0 else 'None'}")

    if len(conflict_spikes) > 0 and len(price_moves) > 0:
        lead_days = (price_moves['date'].iloc[0] - conflict_spikes['date'].iloc[0]).days
        print(f"  >>> GDELT lead time: {lead_days} days")

    # Summary stats
    print(f"  WTI range: ${sub['wti_close'].min():.1f} ~ ${sub['wti_close'].max():.1f}")
    print(f"  Avg conflict intensity: {sub['conflict_intensity_7d'].mean():.4f} (overall avg: {df['conflict_intensity_7d'].mean():.4f})")
    print(f"  Max conflict z-score: {sub['conflict_zscore'].max():.2f}")


# ── Figure 1: Iran-US conflict timeline ───────────────────
print("\n\nGenerating figures...")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Focus on Iran-US conflict period
iran_mask = (df['date'] >= '2025-06-01') & (df['date'] <= '2026-03-13')
iran = df[iran_mask].copy()

# Panel A: WTI price
ax = axes[0]
ax.plot(iran['date'], iran['wti_close'], color='#1f77b4', linewidth=1.5)
ax.fill_between(iran['date'], iran['wti_close'], alpha=0.1, color='#1f77b4')
ax.set_ylabel('WTI Close ($)')
ax.set_title('Panel A: WTI Crude Oil Price')
ax.axvline(pd.Timestamp('2026-03-02'), color='red', linestyle='--', alpha=0.7, label='Price surge onset')
ax.legend(loc='upper left', frameon=False)

# Panel B: GDELT conflict intensity
ax = axes[1]
ax.plot(iran['date'], iran['conflict_intensity_7d'], color='#d62728', linewidth=1.2, label='7-day MA')
ax.plot(iran['date'], iran['conflict_intensity_21d'], color='#ff7f0e', linewidth=1, alpha=0.7, label='21-day MA')
ax.axhline(df['conflict_intensity_7d'].quantile(0.95), color='gray', linestyle=':', alpha=0.5, label='95th percentile')
ax.set_ylabel('Conflict Intensity')
ax.set_title('Panel B: GDELT Oil-Country Conflict Intensity')
ax.axvline(pd.Timestamp('2026-03-02'), color='red', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=False)

# Panel C: Conflict z-score
ax = axes[2]
ax.bar(iran['date'], iran['conflict_zscore'], color=iran['conflict_zscore'].apply(
    lambda x: '#d62728' if x > 2 else '#ff7f0e' if x > 1 else '#2ca02c'), alpha=0.7, width=1)
ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='Alert threshold (z=2)')
ax.axhline(1, color='orange', linestyle='--', alpha=0.3, label='Warning threshold (z=1)')
ax.set_ylabel('Conflict Z-Score')
ax.set_title('Panel C: GDELT Conflict Anomaly Score')
ax.axvline(pd.Timestamp('2026-03-02'), color='red', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=False)

# Panel D: Realized volatility
ax = axes[3]
ax.plot(iran['date'], iran['realized_vol'], color='#9467bd', linewidth=1.2)
ax.axhline(vol_threshold, color='gray', linestyle=':', alpha=0.5, label=f'High-vol threshold ({vol_threshold:.2f})')
ax.set_ylabel('Realized Volatility (ann.)')
ax.set_title('Panel D: WTI Realized Volatility (20-day)')
ax.axvline(pd.Timestamp('2026-03-02'), color='red', linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=False)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/iran_us_conflict_timeline.pdf', bbox_inches='tight')
plt.savefig('figures/iran_us_conflict_timeline.png', bbox_inches='tight')
print("Saved: figures/iran_us_conflict_timeline.pdf")

# ── Figure 2: Cross-correlation plot ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lags, ccf, color=['#d62728' if l > 0 else '#1f77b4' for l in lags], alpha=0.7, width=0.8)
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(1.96/np.sqrt(len(x)), color='gray', linestyle='--', alpha=0.5, label='95% CI')
ax.axhline(-1.96/np.sqrt(len(x)), color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Lag (days, positive = GDELT leads)')
ax.set_ylabel('Cross-correlation')
ax.set_title('Cross-Correlation: GDELT Conflict Intensity → WTI Realized Volatility')
ax.annotate(f'Peak: lag={peak_lag}d, r={peak_corr:.3f}',
            xy=(peak_lag, peak_corr), xytext=(peak_lag+3, peak_corr+0.02),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('figures/cross_correlation.pdf', bbox_inches='tight')
plt.savefig('figures/cross_correlation.png', bbox_inches='tight')
print("Saved: figures/cross_correlation.pdf")

# ── Figure 3: Multi-episode comparison ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (event_name, (start, end)) in enumerate(major_events.items()):
    ax = axes[idx // 2][idx % 2]
    mask = (df['date'] >= start) & (df['date'] <= end)
    if mask.sum() == 0:
        ax.set_title(f'{event_name}: No data')
        continue
    sub = df[mask]

    # Dual axis
    ax2 = ax.twinx()

    l1, = ax.plot(sub['date'], sub['wti_close'], color='#1f77b4', linewidth=1.5, label='WTI Price')
    l2, = ax2.plot(sub['date'], sub['conflict_intensity_7d'], color='#d62728', linewidth=1.2, label='Conflict (7d)')

    ax.set_title(event_name)
    ax.set_ylabel('WTI ($)', color='#1f77b4')
    ax2.set_ylabel('Conflict Intensity', color='#d62728')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(handles=[l1, l2], loc='upper left', frameon=False)

plt.suptitle('GDELT Conflict Intensity vs. WTI Price: Four Major Episodes', fontsize=14)
plt.tight_layout()
plt.savefig('figures/multi_episode_comparison.pdf', bbox_inches='tight')
plt.savefig('figures/multi_episode_comparison.png', bbox_inches='tight')
print("Saved: figures/multi_episode_comparison.pdf")

print("\n\nDone.")
