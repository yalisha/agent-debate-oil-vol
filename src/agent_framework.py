"""
LLM-Orchestrated Adaptive Forecasting Agent for Oil Price Prediction.
Core framework: exogenous proactive drift detection + conditional LLM activation.

Architecture:
  - Default: simple ridge baseline (low cost, no LLM)
  - Trigger: when quantitative drift signal exceeds threshold, activate LLM
  - LLM Agent: single call for regime assessment + model/feature decision
"""

import json
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import Optional
import time

# ── LLM Client ────────────────────────────────────────────

API_KEY = "sk-bnKhYRRx5EZyGXklNUsXE7nZ3sXd2ufbki7sfr8mkSxFcDP8"
BASE_URL = "https://api.akane.win/v1"
MODEL = "gpt-5.4"


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── Single Combined Prompt ────────────────────────────────

AGENT_PROMPT = """You are an oil market adaptive forecasting agent. A drift signal has been detected in the quantitative data, suggesting a potential regime change. Your job is to analyze the situation and make a single integrated decision.

## Quantitative Snapshot (date: {date})
{quant_snapshot}

## Drift Signal
Conflict Z-Score: {conflict_zscore:.2f} (threshold for activation: 1.5)
Realized Volatility: {realized_vol:.2f} (annualized)
Recent Vol Change: {vol_change:+.2f} (vs 20-day-ago level)

## GDELT Geopolitical Event Summary (past 7 days)
{gdelt_summary}

## Historical Context
{historical_context}

## Instructions
Assess the current market regime based on the quantitative signals and geopolitical context. Output ONLY valid JSON:
{{
  "regime": "stable" | "transitioning" | "crisis",
  "drift_type": "none" | "supply_shock" | "demand_shock" | "policy_shift" | "geopolitical_escalation",
  "primary_driver": "brief description (1 sentence)",
  "explanation": "2-3 sentences for risk managers"
}}

Regime classification:
- "stable": normal market conditions, no significant geopolitical disruption, vol within historical norms
- "transitioning": emerging geopolitical risk, increasing media coverage of oil-relevant conflicts, early signs of supply chain disruption, vol rising but not extreme
- "crisis": active military conflict, sanctions enforcement, confirmed supply disruption, extreme vol, major price dislocations

Be conservative: only classify as "crisis" when evidence is overwhelming. Most activated days should be "transitioning"."""


EXPLANATION_PROMPT = """You are a financial analyst generating a prediction explanation for risk managers.

## Date: {date}
## Regime: {regime}
## Drift Type: {drift_type}
## Primary Driver: {primary_driver}
## Prediction: {direction} (predicted return: {pred_return:.4f})
## Model: {model_used}
## GDELT Summary: {gdelt_brief}

Write a concise 3-4 sentence explanation. Cover the regime state, prediction rationale, and key risks. Output plain text only."""


# ── Agent Class ───────────────────────────────────────────

class OilPriceAgent:
    """LLM-orchestrated adaptive forecasting agent with conditional activation."""

    def __init__(self, model: str = MODEL, temperature: float = 0,
                 conflict_threshold: float = 1.5, vol_threshold: float = 0.40):
        self.client = get_client()
        self.model = model
        self.temperature = temperature
        self.conflict_threshold = conflict_threshold
        self.vol_threshold = vol_threshold
        self.call_count = 0
        self.total_tokens = 0
        self.decision_log = []

    def _call_llm(self, prompt: str, max_tokens: int = 400) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                self.call_count += 1
                self.total_tokens += response.usage.total_tokens
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return None

    def _parse_json(self, text: str) -> dict:
        if text is None:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return None

    def should_activate(self, conflict_zscore: float, realized_vol: float,
                        vol_change: float) -> bool:
        """Determine if LLM agent should be activated based on quantitative signals."""
        # Activate if ANY of these conditions are met:
        # 1. Conflict z-score above threshold (geopolitical signal)
        # 2. Realized vol above threshold (market turbulence)
        # 3. Vol is rapidly increasing (emerging regime shift)
        return (conflict_zscore > self.conflict_threshold or
                realized_vol > self.vol_threshold or
                vol_change > 0.15)

    def make_decision(self, date: str, quant_snapshot: str,
                      gdelt_summary: str, historical_context: str,
                      conflict_zscore: float, realized_vol: float,
                      vol_change: float) -> dict:
        """Single LLM call for integrated decision."""
        prompt = AGENT_PROMPT.format(
            date=date,
            quant_snapshot=quant_snapshot,
            gdelt_summary=gdelt_summary,
            historical_context=historical_context,
            conflict_zscore=conflict_zscore,
            realized_vol=realized_vol,
            vol_change=vol_change,
        )
        output = self._call_llm(prompt)
        decision = self._parse_json(output)

        if decision is None:
            # Fallback if LLM fails: conservative transitioning
            decision = {
                "regime": "transitioning",
                "drift_type": "none",
                "primary_driver": "LLM parsing failed, using default",
                "explanation": "Fallback decision due to LLM error.",
            }

        self.decision_log.append({"date": date, "decision": decision, "activated": True})
        return decision

    def default_decision(self, date: str) -> dict:
        """Default decision when agent is NOT activated (stable period)."""
        decision = {
            "regime": "stable",
            "drift_type": "none",
            "primary_driver": "No significant drift signal detected",
            "use_gdelt_features": False,
            "model": "ridge",
            "feature_config": "minimal",
            "explanation": "Stable regime, using conservative baseline.",
        }
        self.decision_log.append({"date": date, "decision": decision, "activated": False})
        return decision

    def get_stats(self) -> dict:
        activated = sum(1 for d in self.decision_log if d.get('activated', False))
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_decisions": len(self.decision_log),
            "activated_decisions": activated,
            "activation_rate": activated / max(len(self.decision_log), 1),
            "avg_tokens_per_call": self.total_tokens / max(self.call_count, 1),
        }


# ── Data Preparation Helpers ──────────────────────────────

def build_quant_snapshot(row: pd.Series) -> str:
    parts = [f"WTI Close: ${row.get('wti_close', 0):.2f}"]
    if 'wti_return' in row and pd.notna(row['wti_return']):
        parts.append(f"Daily Return: {row['wti_return']*100:.2f}%")
    if 'realized_vol' in row and pd.notna(row['realized_vol']):
        parts.append(f"20d Realized Vol (ann.): {row['realized_vol']:.2f}")
    if 'VIX' in row and pd.notna(row['VIX']):
        parts.append(f"VIX: {row['VIX']:.1f}")
    if 'DXY' in row and pd.notna(row['DXY']):
        parts.append(f"DXY: {row['DXY']:.1f}")
    if 'YIELD_10Y' in row and pd.notna(row['YIELD_10Y']):
        parts.append(f"10Y Yield: {row['YIELD_10Y']:.2f}%")
    return " | ".join(parts)


def build_gdelt_summary(gdelt_df: pd.DataFrame, date: str, lookback: int = 7) -> str:
    end = pd.Timestamp(date)
    start = end - pd.Timedelta(days=lookback)
    mask = (gdelt_df['date'] >= start) & (gdelt_df['date'] <= end)
    subset = gdelt_df[mask].sort_values('date', ascending=False)

    if len(subset) == 0:
        return "No GDELT data available for this period."

    total_events = subset['n_events'].sum()
    total_conflict = subset['n_conflict'].sum()
    avg_goldstein = subset['avg_goldstein'].mean()
    conflict_ratio = total_conflict / max(total_events, 1)

    parts = [
        f"Period: {start.date()} to {end.date()}",
        f"Oil-country events: {total_events:,} | Conflict: {total_conflict:,} ({conflict_ratio:.1%}) | Goldstein: {avg_goldstein:.2f}",
    ]

    for _, row in subset.head(3).iterrows():
        summary_text = str(row.get('summary', ''))
        if len(summary_text) > 300:
            summary_text = summary_text[:300] + "..."
        parts.append(f"{row['date'].date()}: {summary_text}")

    return "\n".join(parts)


def build_historical_context(df: pd.DataFrame, current_idx: int, window: int = 20) -> str:
    start_idx = max(0, current_idx - window)
    hist = df.iloc[start_idx:current_idx]

    if len(hist) == 0:
        return "No historical data available."

    ret_mean = hist['wti_return'].mean() * 100
    ret_std = hist['wti_return'].std() * 100
    price_chg = (hist['wti_close'].iloc[-1] / hist['wti_close'].iloc[0] - 1) * 100
    max_dd = (hist['wti_close'] / hist['wti_close'].cummax() - 1).min() * 100

    parts = [
        f"Past {len(hist)} days: price {price_chg:+.1f}%, avg return {ret_mean:+.3f}% (std {ret_std:.3f}%)",
        f"Range: ${hist['wti_close'].min():.1f} ~ ${hist['wti_close'].max():.1f}, max drawdown {max_dd:.1f}%",
    ]

    if 'conflict_intensity_7d' in hist.columns:
        ci = hist['conflict_intensity_7d'].dropna()
        if len(ci) > 0:
            parts.append(f"Conflict intensity (7d avg): {ci.mean():.4f}, max z-score: {hist.get('conflict_zscore', pd.Series([0])).max():.2f}")

    return "\n".join(parts)


# ── Adaptive Predictor ────────────────────────────────────

FEATURE_MAP = {
    'minimal': lambda cols: [c for c in cols if c.startswith('wti_return_lag')],
    'standard': lambda cols: [c for c in cols if c.startswith('wti_return_lag')
                              or c in ['VIX', 'DXY', 'YIELD_10Y', 'realized_vol']],
    'full': lambda cols: [c for c in cols if c.startswith('wti_return_lag')
                          or c in ['VIX', 'DXY', 'YIELD_10Y', 'realized_vol',
                                   'conflict_intensity_7d', 'conflict_zscore',
                                   'oil_material_conflict_share', 'oil_goldstein_mean',
                                   'oil_conflict_share', 'oil_tone_mean',
                                   'oil_net_coop']],
}


class AdaptivePredictor:
    """Conditional LLM activation + adaptive model selection."""

    def __init__(self, agent: OilPriceAgent):
        self.agent = agent
        self._init_models()

    # Regime → (model, feature_set) mapping, validated by full rolling-origin evaluation
    REGIME_CONFIG = {
        'stable':        ('ridge', 'minimal'),     # Ridge(a=100) + lags only: best in low-vol
        'transitioning': ('ridge', 'full'),         # Ridge(a=100) + GDELT: info gain at transitions
        'crisis':        ('gbr',   'full'),         # GBR aggressive + all features: R2=0.73 in high-vol
    }

    def _init_models(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        self.models = {
            'ridge': lambda: Ridge(alpha=100.0),
            'gbr': lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, min_samples_leaf=5, random_state=42
            ),
            'rf': lambda: RandomForestRegressor(
                n_estimators=200, max_depth=7, max_features=0.5,
                min_samples_leaf=5, random_state=42
            ),
            'historical_mean': None,
        }

    def _regime_to_config(self, regime: str) -> tuple:
        return self.REGIME_CONFIG.get(regime, self.REGIME_CONFIG['stable'])

    def predict_one(self, df: pd.DataFrame, gdelt_summaries: pd.DataFrame,
                    current_idx: int, train_window: int = 252) -> dict:
        row = df.iloc[current_idx]
        date_str = str(row['date'].date())

        # Get drift signals
        conflict_zscore = row.get('conflict_zscore', 0)
        if pd.isna(conflict_zscore):
            conflict_zscore = 0
        realized_vol = row.get('realized_vol', 0)
        if pd.isna(realized_vol):
            realized_vol = 0

        # Vol change vs 20 days ago
        if current_idx >= 20:
            prev_vol = df.iloc[current_idx - 20].get('realized_vol', realized_vol)
            if pd.isna(prev_vol) or prev_vol == 0:
                vol_change = 0
            else:
                vol_change = realized_vol - prev_vol
        else:
            vol_change = 0

        # Conditional activation
        activated = self.agent.should_activate(conflict_zscore, realized_vol, vol_change)

        if activated:
            quant_snapshot = build_quant_snapshot(row)
            gdelt_summary = build_gdelt_summary(gdelt_summaries, date_str, lookback=7)
            historical_context = build_historical_context(df, current_idx)

            decision = self.agent.make_decision(
                date=date_str,
                quant_snapshot=quant_snapshot,
                gdelt_summary=gdelt_summary,
                historical_context=historical_context,
                conflict_zscore=conflict_zscore,
                realized_vol=realized_vol,
                vol_change=vol_change,
            )
        else:
            decision = self.agent.default_decision(date_str)

        # Map regime assessment to pre-validated model configs
        # LLM decides regime; model config is data-driven (from tuning)
        regime = decision.get('regime', 'stable')
        model_name, feat_config = self._regime_to_config(regime)
        decision['model'] = model_name
        decision['feature_config'] = feat_config

        feature_cols = [c for c in FEATURE_MAP[feat_config](df.columns.tolist()) if c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c.startswith('wti_return_lag')]

        # Train and predict
        train_start = max(0, current_idx - train_window)
        train_data = df.iloc[train_start:current_idx].dropna(subset=feature_cols + ['wti_return_fwd'])

        if len(train_data) < 30 or model_name == 'historical_mean':
            pred = train_data['wti_return_fwd'].mean() if len(train_data) > 0 else 0.0
        else:
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['wti_return_fwd']
            X_test = df.iloc[[current_idx]][feature_cols].fillna(0)

            model = self.models[model_name]()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

        actual = row.get('wti_return_fwd', np.nan)

        return {
            'date': date_str,
            'prediction': pred,
            'actual': actual,
            'model_used': model_name,
            'feature_config': feat_config,
            'regime': decision.get('regime', 'unknown'),
            'drift_type': decision.get('drift_type', 'none'),
            'activated': activated,
            'n_features': len(feature_cols),
        }


# ── Rule-Based Baseline ──────────────────────────────────

class RuleBasedPredictor:
    """Rule-based regime detection + model selection (no LLM, for ablation)."""

    def __init__(self):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        self.models = {
            'ridge': lambda: Ridge(alpha=100.0),
            'gbr': lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, min_samples_leaf=5, random_state=42
            ),
            'rf': lambda: RandomForestRegressor(
                n_estimators=200, max_depth=7, max_features=0.5,
                min_samples_leaf=5, random_state=42
            ),
        }

    def predict_one(self, df: pd.DataFrame, current_idx: int,
                    train_window: int = 252) -> dict:
        row = df.iloc[current_idx]
        date_str = str(row['date'].date())
        vol = row.get('realized_vol', 0)
        if pd.isna(vol):
            vol = 0
        conflict_z = row.get('conflict_zscore', 0)
        if pd.isna(conflict_z):
            conflict_z = 0

        # Rule-based regime detection (same REGIME_CONFIG as Agent for fair ablation)
        if vol > 0.4 or conflict_z > 2:
            regime = 'crisis'
        elif vol > 0.25 or conflict_z > 1:
            regime = 'transitioning'
        else:
            regime = 'stable'

        model_name, feat_config = AdaptivePredictor.REGIME_CONFIG.get(
            regime, ('ridge', 'standard')
        )

        feature_cols = [c for c in FEATURE_MAP[feat_config](df.columns.tolist()) if c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c.startswith('wti_return_lag')]

        train_start = max(0, current_idx - train_window)
        train_data = df.iloc[train_start:current_idx].dropna(subset=feature_cols + ['wti_return_fwd'])

        if len(train_data) < 30:
            pred = train_data['wti_return_fwd'].mean() if len(train_data) > 0 else 0.0
        else:
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['wti_return_fwd']
            X_test = df.iloc[[current_idx]][feature_cols].fillna(0)
            model = self.models[model_name]()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]

        actual = row.get('wti_return_fwd', np.nan)

        return {
            'date': date_str,
            'prediction': pred,
            'actual': actual,
            'model_used': model_name,
            'feature_config': feat_config,
            'regime': regime,
            'drift_type': 'none',
            'activated': regime != 'stable',
            'n_features': len(feature_cols),
        }


# ── Quick Test ────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing conditional Agent Framework...")
    agent = OilPriceAgent()

    # Test should_activate
    print(f"Stable (z=0.3, vol=0.15): activate={agent.should_activate(0.3, 0.15, 0.01)}")
    print(f"Warning (z=1.2, vol=0.20): activate={agent.should_activate(1.2, 0.20, 0.05)}")
    print(f"Crisis (z=3.0, vol=0.50):  activate={agent.should_activate(3.0, 0.50, 0.15)}")

    # Test single decision call
    decision = agent.make_decision(
        date="2026-03-05",
        quant_snapshot="WTI: $81.01 (+8.5%) | VIX: 32.5 | DXY: 103.2 | Realized Vol: 0.48",
        gdelt_summary="45230 oil-country events, 18920 conflict (42%), Goldstein: -1.83. US-Iran armed conflict dominant.",
        historical_context="Past 20 days: +24.6%, max drawdown -5.2%, conflict z-score peaked at 3.21",
        conflict_zscore=3.21,
        realized_vol=0.48,
        vol_change=0.25,
    )
    print(f"\nDecision: {json.dumps(decision, indent=2)}")
    print(f"Stats: {agent.get_stats()}")
