"""
Test agent diversity with optimized prompts + GPT-5 model.

Run 30 days to compare diversity metrics:
  - v2 (haiku, original prompts)  [already have data]
  - v3 (gpt-5, optimized prompts + data separation)

Key changes from v2:
  1. Stronger role enforcement in system prompts
  2. Each agent sees ONLY its specialist data (no common snapshot)
  3. Explicit instruction to trust domain-specific signals over market consensus
  4. Higher temperature (0.5 vs 0.3) for more diverse outputs
  5. Contrarian bias for some agents
"""

import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).parent.parent

# ── API Config ────────────────────────────────────────────
API_KEY = "sk-bnKhYRRx5EZyGXklNUsXE7nZ3sXd2ufbki7sfr8mkSxFcDP8"
BASE_URL = "https://api.akane.win/v1"
MODEL = "gpt-5.4"  # empty responses common but free, retry handles it

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)

AGENTS = ['geopolitical', 'macro_demand', 'monetary', 'supply_opec',
          'technical', 'sentiment', 'cross_market']

# ══════════════════════════════════════════════════════════
# OPTIMIZED AGENT CONFIGS
# ══════════════════════════════════════════════════════════

AGENT_CONFIGS_V3 = {
    "geopolitical": {
        "label": "Geopolitical Risk Analyst",
        "system": """You are a geopolitical risk analyst. You ONLY analyze political and military factors affecting oil volatility. You deliberately ignore price charts and technical patterns.

Your analytical framework:
- Armed conflicts near oil infrastructure or shipping lanes (Strait of Hormuz, Bab el-Mandeb)
- Sanctions on oil-producing nations (Iran, Russia, Venezuela)
- OPEC political dynamics and member state tensions
- Regime instability in major producers
- Pipeline and port disruptions

You have a HAWKISH bias: you tend to see more risk than other analysts, because geopolitical risks are often underpriced. When conflict indicators are elevated, you forecast LARGER upward adjustments than the consensus.

Do NOT mention technical indicators, chart patterns, or statistical models. Your analysis is purely qualitative and geopolitical.""",
        "temperature": 0.5,
    },
    "macro_demand": {
        "label": "Macro Demand Analyst",
        "system": """You are a macroeconomic demand analyst. You ONLY analyze demand-side fundamentals for oil. You deliberately ignore supply-side factors and geopolitics.

Your analytical framework:
- Global GDP growth expectations (US, China, EU)
- Industrial production and manufacturing PMI
- Chinese economic activity (equity indices as proxy)
- Seasonal demand patterns (summer driving, winter heating)
- Demand destruction from high prices vs demand recovery

You have a MEAN-REVERTING bias: you believe oil demand is structurally stable, and volatility shocks from demand tend to be temporary. When prices spike, you expect demand destruction to reduce future volatility.

Do NOT mention geopolitical events, OPEC decisions, or technical patterns. Your analysis is purely macro-fundamental.""",
        "temperature": 0.5,
    },
    "monetary": {
        "label": "Monetary Policy Analyst",
        "system": """You are a monetary policy and financial conditions analyst. You ONLY analyze how monetary policy and dollar movements affect oil volatility. You deliberately ignore physical oil market fundamentals.

Your analytical framework:
- Federal Reserve policy stance (rate decisions, forward guidance)
- Real interest rates and yield curve shape
- US dollar strength (DXY) and its inverse relationship with oil
- Global liquidity conditions
- Credit spreads as risk appetite indicator

You have a DOVISH-BULLISH bias: you believe loose monetary conditions systematically suppress volatility (via liquidity), and tightening cycles increase it. You weight dollar movements heavily.

Do NOT mention supply/demand fundamentals, geopolitics, or technical patterns. Your analysis is purely financial-channel.""",
        "temperature": 0.5,
    },
    "supply_opec": {
        "label": "OPEC & Supply Analyst",
        "system": """You are a physical oil market analyst specializing in supply dynamics. You ONLY analyze supply-side factors. You deliberately ignore demand-side macro data and financial conditions.

Your analytical framework:
- OPEC+ production quotas and compliance rates
- Spare capacity utilization
- Non-OPEC supply growth (US shale, offshore)
- Inventory draw/build signals from price momentum
- Physical market tightness (backwardation/contango implied by price trends)

You have a SUPPLY-FOCUSED bias: you believe supply disruptions are the dominant driver of oil volatility, and demand changes are secondary. When short-term price momentum is strongly negative, you interpret it as oversupply and forecast LOWER volatility.

Do NOT mention interest rates, geopolitics, or technical indicators. Your analysis is purely supply-fundamental.""",
        "temperature": 0.5,
    },
    "technical": {
        "label": "Quantitative / Statistical Analyst",
        "system": """You are a pure quantitative analyst. You ONLY use statistical patterns and numerical data. You deliberately ignore all narrative and qualitative information.

Your analytical framework:
- Volatility clustering (GARCH-type): high vol today predicts high vol tomorrow
- Mean reversion: extreme vol tends to revert toward long-run average
- Vol-of-vol: acceleration or deceleration in volatility changes
- Regime detection: probability of being in high/low volatility state
- Historical vol ratios (short-term vs long-term)

You have a CONTRARIAN bias: when other analysts are all forecasting in the same direction, you are skeptical. You trust your statistical models more than narratives. If 20d vol is well above 60d vol, you forecast DOWNWARD mean reversion even if the news is scary.

Do NOT mention news events, geopolitics, or supply/demand narratives. Your analysis is purely statistical.""",
        "temperature": 0.4,
    },
    "sentiment": {
        "label": "News Sentiment & Attention Analyst",
        "system": """You are a media sentiment analyst. You ONLY analyze information flow and attention dynamics. You deliberately ignore fundamental analysis.

Your analytical framework:
- News volume: more coverage = more uncertainty = higher volatility
- Media tone shifts: sudden negativity spikes as early warning
- Attention clustering: when multiple crisis narratives coincide
- Narrative fatigue: prolonged negative coverage eventually loses impact
- Cross-media consistency: are different sources telling the same story?

You have an OVERREACTION bias: you believe markets overreact to sudden sentiment shifts in the short term, then correct. When news tone suddenly drops, you forecast a TEMPORARY volatility spike followed by normalization.

Do NOT mention interest rates, supply fundamentals, or statistical patterns. Your analysis is purely sentiment-driven.""",
        "temperature": 0.6,
    },
    "cross_market": {
        "label": "Cross-Market Contagion Analyst",
        "system": """You are a cross-asset contagion analyst. You ONLY analyze spillover effects from other markets to oil. You deliberately ignore oil-specific fundamentals.

Your analytical framework:
- VIX (equity fear gauge) as leading indicator for commodity volatility
- Equity-commodity correlation regimes (risk-on vs risk-off)
- Credit spreads as systemic stress indicator
- Currency stress (DXY, EM currencies)
- Flight-to-quality flows (gold, treasuries)

You have a CONTAGION bias: you believe cross-market stress is the most underappreciated driver of oil volatility. When VIX spikes, you forecast LARGER oil vol increases than other analysts expect, because contagion effects are nonlinear.

Do NOT mention oil supply/demand fundamentals or geopolitics directly. Your analysis is purely about cross-market transmission.""",
        "temperature": 0.5,
    },
}


# ══════════════════════════════════════════════════════════
# DATA PREPARATION (separated by agent, minimal overlap)
# ══════════════════════════════════════════════════════════

def prepare_agent_data_v3(row, gdelt_row=None, horizon=20):
    """Each agent gets ONLY its domain-specific data. No common snapshot."""
    data = {}
    vol_col = f"wti_vol_{horizon}d"
    persistence = row.get(vol_col, 0.25) if pd.notna(row.get(vol_col)) else 0.25

    # Geopolitical: GDELT conflict data only
    geo_parts = []
    if gdelt_row is not None:
        geo_parts.append(f"Oil-region events today: {gdelt_row.get('n_events_oil', 0):.0f}")
        geo_parts.append(f"Conflict events: {gdelt_row.get('oil_n_conflict_events', 0):.0f}")
        geo_parts.append(f"Conflict share of all events: {gdelt_row.get('oil_conflict_share', 0):.1%}")
        geo_parts.append(f"Material conflict share: {gdelt_row.get('oil_material_conflict_share', 0):.1%}")
        geo_parts.append(f"Cooperation-conflict balance (Goldstein): {gdelt_row.get('oil_goldstein_mean', 0):.2f}")
        # Add verbal vs material breakdown
        geo_parts.append(f"Verbal cooperation events: {gdelt_row.get('oil_verbal_coop', 0):.0f}")
        geo_parts.append(f"Material conflict events: {gdelt_row.get('oil_material_conflict', 0):.0f}")
        geo_parts.append(f"Net cooperation index: {gdelt_row.get('oil_net_coop', 0):.3f}")
    data["geopolitical"] = "\n".join(geo_parts) if geo_parts else "No conflict data available today."

    # Macro demand: Chinese equity (demand proxy) + momentum (demand trend)
    macro_parts = []
    if pd.notna(row.get('SHANGHAI_COMPOSITE')):
        macro_parts.append(f"Shanghai Composite index: {row['SHANGHAI_COMPOSITE']:.1f}")
    if pd.notna(row.get('wti_mom_20d')):
        macro_parts.append(f"Oil price 20-day momentum: {row['wti_mom_20d']:.4f}")
    if pd.notna(row.get('wti_mom_5d')):
        macro_parts.append(f"Oil price 5-day momentum: {row['wti_mom_5d']:.4f}")
    # Add price level for demand destruction assessment
    if pd.notna(row.get('wti_price')):
        macro_parts.append(f"Current WTI price: ${row['wti_price']:.2f}")
    data["macro_demand"] = "\n".join(macro_parts) if macro_parts else "No macro data available."

    # Monetary: rates, yields, DXY (no oil price)
    mon_parts = []
    for col, label in [('FED_FUNDS', 'Fed Funds rate'),
                       ('YIELD_2Y', '2-year Treasury yield'),
                       ('YIELD_10Y', '10-year Treasury yield'),
                       ('REAL_YIELD_10Y', '10-year real yield'),
                       ('YIELD_SPREAD_10Y_2Y', 'Yield curve spread (10Y-2Y)')]:
        if pd.notna(row.get(col)):
            mon_parts.append(f"{label}: {row[col]:.2f}%")
    if pd.notna(row.get('DXY')):
        mon_parts.append(f"Dollar index (DXY): {row['DXY']:.1f}")
    if pd.notna(row.get('dxy_return')):
        mon_parts.append(f"Dollar daily change: {row['dxy_return'] * 100:+.2f}%")
    if pd.notna(row.get('spread_change')):
        mon_parts.append(f"Yield spread daily change: {row['spread_change']:+.3f}")
    data["monetary"] = "\n".join(mon_parts) if mon_parts else "No monetary data available."

    # Supply/OPEC: price + momentum (physical market signals, no vol data)
    supply_parts = []
    if pd.notna(row.get('wti_price')):
        supply_parts.append(f"WTI crude price: ${row['wti_price']:.2f}")
    if pd.notna(row.get('wti_return')):
        supply_parts.append(f"Daily price change: {row['wti_return'] * 100:+.2f}%")
    if pd.notna(row.get('wti_mom_5d')):
        supply_parts.append(f"5-day price momentum: {row['wti_mom_5d']:.4f}")
    if pd.notna(row.get('wti_mom_20d')):
        supply_parts.append(f"20-day price momentum: {row['wti_mom_20d']:.4f}")
    data["supply_opec"] = "\n".join(supply_parts) if supply_parts else "No supply data available."

    # Technical: vol data only (no price, no news)
    tech_parts = []
    if pd.notna(row.get(vol_col)):
        tech_parts.append(f"{horizon}-day realized volatility: {row[vol_col]:.4f}")
    if pd.notna(row.get('wti_vol_5d')):
        tech_parts.append(f"5-day realized volatility: {row['wti_vol_5d']:.4f}")
    if pd.notna(row.get('wti_vol_60d')):
        tech_parts.append(f"60-day realized volatility: {row['wti_vol_60d']:.4f}")
    # Add vol ratios for regime detection
    v20 = row.get(vol_col, None)
    v60 = row.get('wti_vol_60d', None)
    if pd.notna(v20) and pd.notna(v60) and v60 > 0:
        tech_parts.append(f"Short/long vol ratio (20d/60d): {v20/v60:.2f}")
    if pd.notna(row.get('wti_return')):
        tech_parts.append(f"Latest daily return: {row['wti_return'] * 100:+.2f}%")
    data["technical"] = "\n".join(tech_parts) if tech_parts else "No technical data available."

    # Sentiment: GDELT tone/volume only (no conflict data)
    sent_parts = []
    if gdelt_row is not None:
        sent_parts.append(f"Oil news volume: {gdelt_row.get('oil_mentions_sum', 0):.0f} mentions")
        sent_parts.append(f"Oil news article count: {gdelt_row.get('oil_articles_sum', 0):.0f}")
        sent_parts.append(f"Oil media tone (positive=cooperative): {gdelt_row.get('oil_tone_mean', 0):.2f}")
        sent_parts.append(f"Tone volatility: {gdelt_row.get('oil_tone_std', 0):.2f}")
        sent_parts.append(f"Global media tone: {gdelt_row.get('global_tone_mean', 0):.2f}")
    data["sentiment"] = "\n".join(sent_parts) if sent_parts else "No sentiment data available."

    # Cross-market: VIX + yield spread change (no oil-specific data)
    cross_parts = []
    if pd.notna(row.get('VIX')):
        cross_parts.append(f"VIX (equity volatility): {row['VIX']:.1f}")
    if pd.notna(row.get('vix_change')):
        cross_parts.append(f"VIX daily change: {row['vix_change']:+.2f}")
    if pd.notna(row.get('spread_change')):
        cross_parts.append(f"Credit/yield spread change: {row['spread_change']:+.3f}")
    if pd.notna(row.get('SHANGHAI_COMPOSITE')):
        cross_parts.append(f"Shanghai Composite: {row['SHANGHAI_COMPOSITE']:.1f}")
    data["cross_market"] = "\n".join(cross_parts) if cross_parts else "No cross-market data available."

    return data


# ══════════════════════════════════════════════════════════
# OPTIMIZED PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════

OPINION_SPEC_V3 = """Yesterday's realized volatility (annualized) is {persistence_vol:.4f}.
Predict how much volatility will CHANGE from this baseline, based SOLELY on your domain indicators.

Output ONLY valid JSON:
{{
  "vol_adjustment": <float between -0.15 and +0.15>,
  "direction": "up" | "down" | "stable",
  "confidence": <float 0-1, be honest about uncertainty>,
  "evidence": [<string: specific data point from YOUR domain>, <string>, ...],
  "revision_reason": "<why you changed from prior round, empty if round 1>"
}}

CRITICAL: Your adjustment MUST be driven by YOUR specialist data above, not generic market reasoning. Different specialists seeing different data should reach different conclusions."""

ROUND1_V3 = """## Date: {date}

## Your Domain-Specific Indicators
{agent_specific_data}

Analyze ONLY the indicators above through the lens of your expertise.
""" + OPINION_SPEC_V3

ROUND_N_V3 = """## Date: {date}

## Your Domain-Specific Indicators
{agent_specific_data}

## Other Analysts' Forecasts (Round {prev_round})
{other_opinions}

## Your Previous Forecast (Round {prev_round})
{own_previous}

Consider the other views, but maintain your position if your domain evidence supports it.
Conforming without domain-specific justification is penalized.
""" + OPINION_SPEC_V3


# ══════════════════════════════════════════════════════════
# DEBATE ENGINE (SIMPLIFIED FOR TESTING)
# ══════════════════════════════════════════════════════════

def call_llm(agent_id, system_prompt, user_prompt, temperature=0.5):
    for attempt in range(15):  # more retries for empty responses
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=500,
                timeout=120.0,
            )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                # Empty response from 5.4 is common and free, just retry
                time.sleep(0.5)
                continue
            content = content.strip()
            # Extract JSON
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            # Try to find JSON in content
            start = content.find('{')
            end = content.rfind('}')
            if start >= 0 and end > start:
                content = content[start:end+1]
            parsed = json.loads(content)
            return {
                'vol_adjustment': float(parsed.get('vol_adjustment', 0)),
                'direction': parsed.get('direction', 'stable'),
                'confidence': float(parsed.get('confidence', 0.5)),
                'evidence': parsed.get('evidence', []),
                'revision_reason': parsed.get('revision_reason', ''),
            }
        except Exception as e:
            if '429' in str(e):
                time.sleep(5)
            elif 'timeout' in str(e).lower():
                time.sleep(2)
            else:
                time.sleep(1)
    print(f"  WARNING: {agent_id} failed after 15 attempts")
    return None


def run_debate_one_day(date, market_row, gdelt_row, persistence_vol, n_rounds=3):
    """Run 3-round debate for one day, return per-agent per-round results."""
    agent_data = prepare_agent_data_v3(market_row, gdelt_row)
    results = {}

    for r in range(1, n_rounds + 1):
        round_results = {}

        def call_agent(agent_id):
            config = AGENT_CONFIGS_V3[agent_id]
            temp = config.get('temperature', 0.5)

            if r == 1:
                prompt = ROUND1_V3.format(
                    date=date,
                    agent_specific_data=agent_data[agent_id],
                    persistence_vol=persistence_vol,
                )
            else:
                # Compile other opinions from previous round
                other_ops = []
                for a2 in AGENTS:
                    if a2 != agent_id and (a2, r - 1) in results:
                        prev = results[(a2, r - 1)]
                        sign = "+" if prev['vol_adjustment'] >= 0 else ""
                        other_ops.append(
                            f"[{a2}] adj={sign}{prev['vol_adjustment']:.4f} "
                            f"({prev['direction']}), conf={prev['confidence']:.2f}"
                        )
                own_prev = results.get((agent_id, r - 1), {})
                own_str = json.dumps(own_prev, indent=2) if own_prev else "No previous forecast."

                prompt = ROUND_N_V3.format(
                    date=date,
                    agent_specific_data=agent_data[agent_id],
                    prev_round=r - 1,
                    other_opinions="\n".join(other_ops),
                    own_previous=own_str,
                    persistence_vol=persistence_vol,
                )

            return agent_id, call_llm(agent_id, config['system'], prompt, temp)

        # Parallel calls for this round
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = [pool.submit(call_agent, a) for a in AGENTS]
            for f in futures:
                agent_id, result = f.result()
                if result:
                    results[(agent_id, r)] = result
                    round_results[agent_id] = result

    return results


# ══════════════════════════════════════════════════════════
# MAIN TEST
# ══════════════════════════════════════════════════════════

def main():
    # Load data
    oil = pd.read_csv(BASE / 'data' / 'oil_macro_daily.csv')
    oil['date'] = pd.to_datetime(oil['date']).dt.strftime('%Y-%m-%d')

    gdelt = pd.read_csv(BASE / 'data' / 'gdelt_daily_features.csv')
    gdelt['date'] = pd.to_datetime(gdelt['date']).dt.strftime('%Y-%m-%d')
    gdelt_map = {r['date']: r for _, r in gdelt.iterrows()}

    # Use recent dates for testing (2024-01 to 2024-02, ~30 days)
    test_dates = oil[(oil['date'] >= '2024-01-02') & (oil['date'] <= '2024-02-15')]
    test_dates = test_dates[test_dates['wti_vol_20d'].notna()].head(30)

    print(f"Testing {len(test_dates)} days with GPT-5 + optimized prompts")
    print(f"Date range: {test_dates.date.iloc[0]} to {test_dates.date.iloc[-1]}")
    print()

    all_results = []

    for idx, (_, row) in enumerate(test_dates.iterrows()):
        date = row['date']
        persistence_vol = row['wti_vol_20d']
        gdelt_row = gdelt_map.get(date, None)

        t0 = time.time()
        results = run_debate_one_day(date, row, gdelt_row, persistence_vol)
        elapsed = time.time() - t0

        # Extract R1 and R3 adjustments
        r1_adjs = [results.get((a, 1), {}).get('vol_adjustment', 0) for a in AGENTS]
        r3_adjs = [results.get((a, 3), {}).get('vol_adjustment', 0) for a in AGENTS]
        r1_confs = [results.get((a, 1), {}).get('confidence', 0.5) for a in AGENTS]

        # Diversity metrics
        r1_std = np.std(r1_adjs)
        r3_std = np.std(r3_adjs)
        r1_range = max(r1_adjs) - min(r1_adjs)
        n_unique_dir = len(set(results.get((a, 1), {}).get('direction', 'stable') for a in AGENTS))

        day_result = {
            'date': date,
            'persistence_vol': persistence_vol,
            'r1_std': r1_std,
            'r3_std': r3_std,
            'r1_range': r1_range,
            'n_unique_directions_r1': n_unique_dir,
            'r1_mean_adj': np.mean(r1_adjs),
            'r3_mean_adj': np.mean(r3_adjs),
            'r1_mean_conf': np.mean(r1_confs),
        }
        for a in AGENTS:
            day_result[f'adj_r1_{a}'] = results.get((a, 1), {}).get('vol_adjustment', 0)
            day_result[f'adj_r3_{a}'] = results.get((a, 3), {}).get('vol_adjustment', 0)
            day_result[f'conf_r1_{a}'] = results.get((a, 1), {}).get('confidence', 0.5)

        all_results.append(day_result)

        print(f"[{idx+1:2d}/{len(test_dates)}] {date}  "
              f"R1_std={r1_std:.5f} R3_std={r3_std:.5f} "
              f"dirs={n_unique_dir} "
              f"range={r1_range:.4f}  "
              f"{elapsed:.1f}s")

    # Summary comparison with v2 (haiku)
    df_v3 = pd.DataFrame(all_results)

    # Load v2 data for same dates
    v2 = pd.read_csv(BASE / 'results' / 'debate_eval_full_v2_20260328_2014.csv')
    v2 = v2.sort_values('date')
    v2_subset = v2[v2['date'].isin(df_v3['date'].values)]

    print("\n" + "=" * 70)
    print("DIVERSITY COMPARISON: v2 (haiku) vs v3 (gpt-5 + optimized)")
    print("=" * 70)

    if len(v2_subset) > 0:
        v2_r1_stds = []
        v2_r3_stds = []
        for _, vrow in v2_subset.iterrows():
            r1 = [vrow.get(f'adj_r1_{a}', 0) for a in AGENTS]
            r3 = [vrow.get(f'adj_r3_{a}', 0) for a in AGENTS]
            v2_r1_stds.append(np.std(r1))
            v2_r3_stds.append(np.std(r3))

        print(f"\n{'Metric':<35s} {'v2 (haiku)':>12s} {'v3 (gpt-5)':>12s} {'Ratio':>8s}")
        print("─" * 70)

        v2_r1_mean = np.mean(v2_r1_stds)
        v3_r1_mean = df_v3['r1_std'].mean()
        print(f"{'R1 cross-agent std (mean)':<35s} {v2_r1_mean:12.5f} {v3_r1_mean:12.5f} {v3_r1_mean/max(v2_r1_mean,1e-8):8.1f}x")

        v2_r3_mean = np.mean(v2_r3_stds)
        v3_r3_mean = df_v3['r3_std'].mean()
        print(f"{'R3 cross-agent std (mean)':<35s} {v2_r3_mean:12.5f} {v3_r3_mean:12.5f} {v3_r3_mean/max(v2_r3_mean,1e-8):8.1f}x")

        v2_same = sum(1 for s in v2_r1_stds if s == 0)
        v3_same = (df_v3['r1_std'] == 0).sum()
        print(f"{'R1 all-same-adj days':<35s} {v2_same:12d} {v3_same:12d}")

        v3_dirs = df_v3['n_unique_directions_r1'].mean()
        print(f"{'R1 unique directions (mean)':<35s} {'N/A':>12s} {v3_dirs:12.2f}")

        # Pairwise correlations
        v2_r1_corrs = []
        v3_r1_corrs = []
        for i in range(len(AGENTS)):
            for j in range(i+1, len(AGENTS)):
                a, b = AGENTS[i], AGENTS[j]
                if f'adj_r1_{a}' in v2_subset.columns:
                    c = np.corrcoef(v2_subset[f'adj_r1_{a}'].values, v2_subset[f'adj_r1_{b}'].values)[0,1]
                    v2_r1_corrs.append(c)
                c3 = np.corrcoef(df_v3[f'adj_r1_{a}'].values, df_v3[f'adj_r1_{b}'].values)[0,1]
                v3_r1_corrs.append(c3)

        if v2_r1_corrs:
            print(f"{'R1 mean pairwise correlation':<35s} {np.mean(v2_r1_corrs):12.4f} {np.mean(v3_r1_corrs):12.4f}")

    # Save results
    out_path = BASE / 'results' / 'debate_diversity_test_v3.csv'
    df_v3.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
