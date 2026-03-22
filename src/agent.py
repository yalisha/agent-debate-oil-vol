"""
LLM-Orchestrated Forecasting Agent for WTI Crude Oil.

The Agent operates at each prediction time point t:
1. Reads current quantitative indicators and GDELT event summary
2. Extracts structured signals from text (supply, demand, geopolitical risk)
3. Assesses current market regime
4. Selects features and prediction model
5. Generates structured explanation

The LLM (GPT-4o) is the orchestrator, NOT the predictor.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Optional
from openai import OpenAI

CACHE_DIR = "/Users/mac/computerscience/17Agent可解释预测/data/agent_cache"


# ============================================================
# Prompt templates
# ============================================================

SYSTEM_PROMPT = """You are an expert oil market analyst operating as part of a forecasting system.
Your role is to analyze multi-source information at each time point and produce structured assessments.

You will receive:
1. Current market data snapshot (WTI price, VIX, DXY, yield spread, volatility)
2. Recent GDELT geopolitical event summary for oil-producing countries
3. Recent price dynamics (returns, momentum, volatility trend)

Your output must be a valid JSON object with exactly these fields:

{
  "regime_assessment": {
    "regime": "low_volatility" | "high_volatility" | "transition",
    "confidence": 0.0-1.0,
    "primary_driver": "fundamentals" | "geopolitical" | "macro_financial" | "speculative",
    "reasoning": "one sentence"
  },
  "signal_extraction": {
    "supply_signal": {"direction": -1|0|1, "magnitude": 0.0-1.0, "source": "string"},
    "demand_signal": {"direction": -1|0|1, "magnitude": 0.0-1.0, "source": "string"},
    "geopolitical_risk": {"level": 0.0-1.0, "source": "string"},
    "policy_stance": {"direction": -1|0|1, "source": "string"}
  },
  "feature_recommendation": {
    "include_gdelt": true|false,
    "include_macro": true|false,
    "preferred_model": "linear" | "tree" | "all",
    "reasoning": "one sentence"
  },
  "directional_view": {
    "direction": -1|0|1,
    "confidence": 0.0-1.0
  }
}

Rules:
- Be concise. Each reasoning field should be one sentence.
- Base your assessment only on the provided data. Do not invent events.
- If information is insufficient, set confidence to low values.
- direction: -1 = bearish, 0 = neutral, 1 = bullish for oil prices."""


def build_user_prompt(market_data: Dict, gdelt_summary: str, lookback_stats: Dict) -> str:
    """Construct the user prompt for a single prediction time point."""

    prompt = f"""=== Market Snapshot (as of {market_data.get('date', 'N/A')}) ===
WTI Price: ${market_data.get('wti_price', 'N/A'):.2f}
1-day return: {market_data.get('wti_return', 0)*100:.2f}%
5-day momentum: {market_data.get('mom_5d', 0)*100:.2f}%
20-day momentum: {market_data.get('mom_20d', 0)*100:.2f}%
20-day volatility (annualized): {market_data.get('wti_vol_20d', 0)*100:.1f}%
VIX: {market_data.get('VIX', 'N/A'):.1f}
DXY: {market_data.get('DXY', 'N/A'):.2f}
10Y-2Y Spread: {market_data.get('YIELD_SPREAD_10Y_2Y', 'N/A'):.2f}

=== Recent Price Dynamics ===
5-day avg return: {lookback_stats.get('avg_ret_5d', 0)*100:.3f}%
5-day vol ratio (vs 20d): {lookback_stats.get('vol_ratio', 1):.2f}
Max drawdown (20d): {lookback_stats.get('max_dd_20d', 0)*100:.2f}%

=== GDELT Geopolitical Events ===
{gdelt_summary}

Based on the above information, provide your structured assessment as a JSON object."""

    return prompt


class ForecastingAgent:
    """
    LLM-orchestrated forecasting agent.

    At each time point, the agent:
    1. Calls LLM to extract signals and assess regime
    2. Uses the assessment to select features and model
    3. Runs the selected model for prediction
    4. Generates explanation
    """

    def __init__(self, api_key=None, model='gpt-4o', use_cache=True):
        self.client = OpenAI(api_key=api_key or os.environ.get('OPENAI_API_KEY'))
        self.model = model
        self.use_cache = use_cache
        self.cache = {}

        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _call_llm(self, market_data: Dict, gdelt_summary: str,
                  lookback_stats: Dict) -> Dict:
        """Call LLM and parse structured response."""
        user_prompt = build_user_prompt(market_data, gdelt_summary, lookback_stats)

        # Check cache
        if self.use_cache:
            key = self._cache_key(user_prompt)
            cache_path = os.path.join(CACHE_DIR, f"{key}.json")
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    return json.load(f)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=600,
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        # Cache
        if self.use_cache:
            with open(cache_path, 'w') as f:
                json.dump(result, f)

        return result

    def assess(self, date, df, gdelt_summaries, lookback=20):
        """
        Run agent assessment for a given date.

        Parameters
        ----------
        date : datetime-like
            Current prediction date.
        df : DataFrame
            Full dataset (will be filtered to <= date).
        gdelt_summaries : dict
            {date_str: summary_text} lookup.
        lookback : int
            Days of history to compute stats from.

        Returns
        -------
        dict with keys: regime, features, model, signals, explanation
        """
        date = pd.Timestamp(date)
        hist = df[df['date'] <= date].tail(lookback + 1)

        if len(hist) < 5:
            return self._default_assessment()

        current = hist.iloc[-1]

        # Market data snapshot
        market_data = {
            'date': date.strftime('%Y-%m-%d'),
            'wti_price': current.get('wti_price', 0),
            'wti_return': current.get('wti_return', 0),
            'mom_5d': current.get('wti_mom_5d', 0),
            'mom_20d': current.get('wti_mom_20d', 0),
            'wti_vol_20d': current.get('wti_vol_20d', 0),
            'VIX': current.get('VIX', 0),
            'DXY': current.get('DXY', 0),
            'YIELD_SPREAD_10Y_2Y': current.get('YIELD_SPREAD_10Y_2Y', 0),
        }

        # Lookback stats
        returns = hist['wti_return'].dropna()
        lookback_stats = {
            'avg_ret_5d': returns.tail(5).mean() if len(returns) >= 5 else 0,
            'vol_ratio': (returns.tail(5).std() / (returns.tail(20).std() + 1e-8))
                         if len(returns) >= 20 else 1.0,
            'max_dd_20d': returns.tail(20).min() if len(returns) >= 20 else 0,
        }

        # GDELT summary
        date_str = date.strftime('%Y-%m-%d')
        gdelt_summary = gdelt_summaries.get(date_str, 'No GDELT data available for this date.')

        # Call LLM
        try:
            assessment = self._call_llm(market_data, gdelt_summary, lookback_stats)
        except Exception as e:
            print(f"LLM call failed for {date_str}: {e}")
            return self._default_assessment()

        # Parse into actionable decisions
        regime = assessment.get('regime_assessment', {})
        features = assessment.get('feature_recommendation', {})
        signals = assessment.get('signal_extraction', {})
        direction = assessment.get('directional_view', {})

        return {
            'date': date_str,
            'regime': regime.get('regime', 'low_volatility'),
            'regime_confidence': regime.get('confidence', 0.5),
            'primary_driver': regime.get('primary_driver', 'fundamentals'),
            'include_gdelt': features.get('include_gdelt', True),
            'include_macro': features.get('include_macro', True),
            'preferred_model': features.get('preferred_model', 'all'),
            'directional_view': direction.get('direction', 0),
            'directional_confidence': direction.get('confidence', 0.5),
            'supply_direction': signals.get('supply_signal', {}).get('direction', 0),
            'demand_direction': signals.get('demand_signal', {}).get('direction', 0),
            'geopolitical_risk': signals.get('geopolitical_risk', {}).get('level', 0.3),
            'raw_assessment': assessment,
        }

    def _default_assessment(self):
        return {
            'regime': 'low_volatility',
            'regime_confidence': 0.5,
            'primary_driver': 'fundamentals',
            'include_gdelt': True,
            'include_macro': True,
            'preferred_model': 'all',
            'directional_view': 0,
            'directional_confidence': 0.3,
            'supply_direction': 0,
            'demand_direction': 0,
            'geopolitical_risk': 0.3,
            'raw_assessment': {},
        }


class AgentPredictor:
    """
    Agent-enhanced predictor that uses LLM assessments to
    adaptively select features, model, and augment predictions.

    This wraps the baseline models but adapts them based on
    the agent's structured assessment.
    """

    def __init__(self, agent: ForecastingAgent, base_models: Dict):
        self.agent = agent
        self.base_models = base_models

    def predict(self, date, df, gdelt_summaries, features_all, target, train_window=504):
        """
        Make an agent-enhanced prediction for a given date.

        Returns prediction value and full assessment.
        """
        assessment = self.agent.assess(date, df, gdelt_summaries)

        # Select features based on agent recommendation
        selected_features = self._select_features(features_all, assessment)

        # Select model based on agent recommendation
        model_name = self._select_model(assessment)

        # Get training data
        date_ts = pd.Timestamp(date)
        mask = df['date'] < date_ts
        train = df[mask].tail(train_window).dropna(subset=selected_features + [target])

        if len(train) < 50:
            return None, assessment

        X_train = train[selected_features].values
        y_train = train[target].values

        # Current features
        current = df[df['date'] == date_ts]
        if len(current) == 0:
            return None, assessment

        X_test = current[selected_features].values

        # Fit and predict
        model = self.base_models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]

        # Augment prediction with directional view
        directional_adj = assessment['directional_view'] * assessment['directional_confidence'] * 0.001
        y_pred_augmented = y_pred + directional_adj

        assessment['selected_features'] = selected_features
        assessment['selected_model'] = model_name
        assessment['base_prediction'] = y_pred
        assessment['augmented_prediction'] = y_pred_augmented

        return y_pred_augmented, assessment

    def _select_features(self, all_features, assessment):
        """Select feature subset based on agent assessment."""
        base = [f for f in all_features if not f.startswith('oil_') and 'VIX' not in f
                and 'DXY' not in f and 'YIELD' not in f and 'REAL_' not in f]

        selected = list(base)

        if assessment.get('include_macro', True):
            macro = [f for f in all_features if any(k in f for k in ['VIX', 'DXY', 'YIELD', 'REAL_'])]
            selected.extend(macro)

        if assessment.get('include_gdelt', True):
            gdelt = [f for f in all_features if f.startswith('oil_') or f.startswith('n_events')]
            selected.extend(gdelt)

        # Add agent-extracted signals as features
        agent_features = [
            'supply_direction', 'demand_direction',
            'geopolitical_risk', 'directional_view',
        ]

        return selected

    def _select_model(self, assessment):
        """Select model based on agent recommendation."""
        pref = assessment.get('preferred_model', 'all')

        if pref == 'linear':
            return 'Ridge' if 'Ridge' in self.base_models else list(self.base_models.keys())[0]
        elif pref == 'tree':
            for name in ['GBR', 'RF']:
                if name in self.base_models:
                    return name
        # Default: use GBR as it's generally strongest
        return 'GBR' if 'GBR' in self.base_models else list(self.base_models.keys())[0]
