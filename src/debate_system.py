"""
Multi-Agent Debate System for Oil Price Volatility Forecasting
with Accountability Attribution.

Target: next-day realized volatility (not return direction).
Volatility is genuinely predictable (clustering), so attribution
of forecast failures is meaningful.

Architecture:
  - 7 Specialist Analyst Agents with distinct information sets
  - 1 Aggregator
  - Multi-round structured debate
  - Debate Influence Graph
  - Shapley / Myerson value attribution
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
import time
from itertools import combinations

# ── LLM Client ────────────────────────────────────────────

API_KEY = "sk-0cD18Q4iwm4RjqPJtW7JTxSh9orPmEA5hv4l5T2raNdlGa4L"
BASE_URL = "https://api.akane.win/v1"
MODEL = "gpt-5.4"


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)


# ── Data Structures ───────────────────────────────────────

@dataclass
class AgentOpinion:
    """Structured output from an analyst agent in one debate round."""
    agent_id: str
    round_num: int
    vol_adjustment: float    # predicted adjustment to persistence vol (e.g. +0.05 = vol up 5pp)
    direction: str           # "up", "down", "stable"
    confidence: float        # 0-1
    evidence: list           # supporting evidence
    revision_reason: str = ""

    @property
    def vol_forecast(self) -> float:
        """For backward compat: absolute vol = persistence + adjustment."""
        return self.vol_adjustment  # actual absolute computed at aggregation

    def to_prompt_summary(self) -> str:
        sign = "+" if self.vol_adjustment >= 0 else ""
        return (
            f"[{self.agent_id}] adj={sign}{self.vol_adjustment:.4f} ({self.direction}), "
            f"conf={self.confidence:.2f} | {'; '.join(self.evidence[:2])}"
        )


@dataclass
class DebateRecord:
    """Complete record of a debate session at one prediction time point."""
    date: str
    opinions: dict = field(default_factory=dict)  # {(agent_id, round): AgentOpinion}
    final_vol_forecast: float = 0.0
    actual_vol: float = 0.0
    aggregator_weights: dict = field(default_factory=dict)
    influence_edges: list = field(default_factory=list)

    def get_opinion(self, agent_id: str, round_num: int) -> Optional[AgentOpinion]:
        return self.opinions.get((agent_id, round_num))

    def get_round_opinions(self, round_num: int) -> list:
        return [op for key, op in self.opinions.items() if key[1] == round_num]


# ── 7 Agent Definitions ──────────────────────────────────

AGENT_CONFIGS = {
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
    },
    "technical": {
        "label": "Technical / Quantitative Analyst",
        "system": """You are a pure quantitative analyst. You ONLY use statistical patterns and numerical data. You deliberately ignore all narrative and qualitative information.

Your analytical framework:
- Volatility clustering (GARCH-type): high vol today predicts high vol tomorrow
- Mean reversion: extreme vol tends to revert toward long-run average
- Vol-of-vol: acceleration or deceleration in volatility changes
- Regime detection: probability of being in high/low volatility state
- Historical vol ratios (short-term vs long-term)

You have a CONTRARIAN bias: when other analysts are all forecasting in the same direction, you are skeptical. You trust your statistical models more than narratives. If 20d vol is well above 60d vol, you forecast DOWNWARD mean reversion even if the news is scary.

Do NOT mention news events, geopolitics, or supply/demand narratives. Your analysis is purely statistical.""",
    },
    "sentiment": {
        "label": "News Sentiment Analyst",
        "system": """You are a media sentiment analyst. You ONLY analyze information flow and attention dynamics. You deliberately ignore fundamental analysis.

Your analytical framework:
- News volume: more coverage = more uncertainty = higher volatility
- Media tone shifts: sudden negativity spikes as early warning
- Attention clustering: when multiple crisis narratives coincide
- Narrative fatigue: prolonged negative coverage eventually loses impact
- Cross-media consistency: are different sources telling the same story?

You have an OVERREACTION bias: you believe markets overreact to sudden sentiment shifts in the short term, then correct. When news tone suddenly drops, you forecast a TEMPORARY volatility spike followed by normalization.

Do NOT mention interest rates, supply fundamentals, or statistical patterns. Your analysis is purely sentiment-driven.""",
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
    },
}

AGENT_IDS = list(AGENT_CONFIGS.keys())
N_AGENTS = len(AGENT_IDS)

# ── Prompt Templates ──────────────────────────────────────

OPINION_JSON_SPEC = """Yesterday's realized volatility (annualized) is {persistence_vol:.4f}.
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

ROUND1_TEMPLATE = """## Date: {date}

## Your Domain-Specific Indicators
{agent_specific_data}

Analyze ONLY the indicators above through the lens of your expertise.
""" + OPINION_JSON_SPEC

ROUND_N_TEMPLATE = """## Date: {date}

## Your Domain-Specific Indicators
{agent_specific_data}

## Other Analysts' Forecasts (Round {prev_round})
{other_opinions}

## Your Previous Forecast (Round {prev_round})
{own_previous}

Consider the other views, but maintain your position if your domain evidence supports it.
Conforming without domain-specific justification is penalized.
""" + OPINION_JSON_SPEC

AGGREGATOR_TEMPLATE = """You are a chief risk officer synthesizing 7 specialist volatility forecasts.
The analysts debated over {n_rounds} rounds. Synthesize into a single volatility forecast.

## Date: {date}

## Yesterday's Realized Volatility (persistence anchor)
{persistence_vol:.4f} (annualized)

## Final Analyst Forecasts (Round {final_round})
{final_opinions}

## Debate Evolution
{debate_summary}

Important: yesterday's realized vol ({persistence_vol:.4f}) is a strong baseline.
Your forecast should deviate from it only if analyst evidence justifies the deviation.
Weight analysts by confidence and evidence quality. Output ONLY valid JSON:
{{
  "vol_forecast": <float, synthesized annualized volatility>,
  "regime": "low_vol" | "normal" | "elevated" | "crisis",
  "confidence": <float, 0-1>,
  "weights": {{<agent_id>: <float>, ...}},
  "rationale": "<2-3 sentences>"
}}"""

# ── Single-Agent Baseline Prompt ──────────────────────────

SINGLE_AGENT_TEMPLATE = """You are a senior oil market risk analyst with access to all available data.
Forecast tomorrow's oil price volatility.

## Date: {date}

## Yesterday's Realized Volatility: {persistence_vol:.4f} (annualized)

## Market Data
{market_data}

## Geopolitical Data
{geo_data}

## Macro Data
{macro_data}

## Monetary Data
{monetary_data}

## Supply Data
{supply_data}

## Technical Data
{technical_data}

## Sentiment Data
{sentiment_data}

## Cross-Market Data
{cross_market_data}

Output ONLY valid JSON:
{{
  "vol_forecast": <float, predicted next-day annualized volatility>,
  "regime": "low_vol" | "normal" | "elevated" | "crisis",
  "confidence": <float, 0-1>,
  "rationale": "<2-3 sentences>"
}}"""


# ── Debate Engine ─────────────────────────────────────────

class DebateEngine:
    """Orchestrates multi-round debate among 7 analyst agents."""

    def __init__(self, n_rounds: int = 3, model: str = MODEL, temperature: float = 0.5,
                 call_delay: float = 1.0):
        self.client = get_client()
        self.model = model
        self.temperature = temperature
        self.n_rounds = n_rounds
        self.call_delay = call_delay  # seconds between API calls
        self.call_count = 0
        self.total_tokens = 0
        self.fail_count = 0

    def _call_llm(self, system: str, user: str, max_tokens: int = 2000) -> str:
        max_retries = 15  # more retries for gpt-5.4 empty responses
        for attempt in range(max_retries):
            try:
                time.sleep(self.call_delay)  # rate limiting
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                self.call_count += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                content = response.choices[0].message.content
                if not content or not content.strip():
                    # Empty response (common with gpt-5.4, free), retry
                    time.sleep(0.5)
                    continue
                return content
            except Exception as e:
                wait = min(2 ** attempt * 2, 60)
                print(f"    [API retry {attempt+1}/{max_retries}, waiting {wait}s: {str(e)[:80]}]")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    self.fail_count += 1
                    return None

    def _parse_opinion(self, text: str, agent_id: str, round_num: int) -> AgentOpinion:
        if text is None:
            return self._default_opinion(agent_id, round_num, "LLM call failed")
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            data = json.loads(cleaned.strip())
            adj = float(data.get("vol_adjustment", 0.0))
            # Clamp to reasonable range
            adj = max(-0.5, min(0.5, adj))
            return AgentOpinion(
                agent_id=agent_id,
                round_num=round_num,
                vol_adjustment=adj,
                direction=data.get("direction", "stable"),
                confidence=float(data.get("confidence", 0.5)),
                evidence=data.get("evidence", []),
                revision_reason=data.get("revision_reason", ""),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._default_opinion(agent_id, round_num, "Parse failed")

    def _default_opinion(self, agent_id: str, round_num: int, reason: str) -> AgentOpinion:
        return AgentOpinion(
            agent_id=agent_id, round_num=round_num,
            vol_adjustment=0.0, direction="stable", confidence=0.1,
            evidence=[reason], revision_reason=reason if round_num > 1 else "",
        )

    def run_debate(self, date: str, market_data: str,
                   agent_data: dict, actual_vol: float = np.nan,
                   persistence_vol: float = 0.25) -> DebateRecord:
        """Run full multi-round debate, return structured record.
        Agents within the same round are called in parallel.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        record = DebateRecord(date=date, actual_vol=actual_vol)
        self._persistence_vol = persistence_vol  # for aggregator

        def _call_agent(agent_id, config, prompt, round_num):
            raw = self._call_llm(config["system"], prompt)
            return agent_id, self._parse_opinion(raw, agent_id, round_num)

        # Round 1: parallel independent forecasts
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = []
            for agent_id, config in AGENT_CONFIGS.items():
                prompt = ROUND1_TEMPLATE.format(
                    date=date, market_data=market_data,
                    agent_specific_data=agent_data.get(agent_id, "No data available."),
                    persistence_vol=persistence_vol,
                )
                futures.append(pool.submit(_call_agent, agent_id, config, prompt, 1))
            for f in as_completed(futures):
                aid, opinion = f.result()
                record.opinions[(aid, 1)] = opinion

        # Rounds 2..N: parallel within each round (sequential across rounds)
        for r in range(2, self.n_rounds + 1):
            prev_opinions = record.get_round_opinions(r - 1)
            with ThreadPoolExecutor(max_workers=7) as pool:
                futures = []
                for agent_id, config in AGENT_CONFIGS.items():
                    others = [op.to_prompt_summary() for op in prev_opinions if op.agent_id != agent_id]
                    own_prev = record.get_opinion(agent_id, r - 1)
                    prompt = ROUND_N_TEMPLATE.format(
                        date=date, market_data=market_data,
                        agent_specific_data=agent_data.get(agent_id, ""),
                        prev_round=r - 1,
                        other_opinions="\n".join(others),
                        own_previous=own_prev.to_prompt_summary() if own_prev else "N/A",
                        persistence_vol=persistence_vol,
                    )
                    futures.append(pool.submit(_call_agent, agent_id, config, prompt, r))
                for f in as_completed(futures):
                    aid, opinion = f.result()
                    record.opinions[(aid, r)] = opinion

        # Influence edges
        record.influence_edges = self._compute_influence_edges(record)

        # Aggregation
        record.final_vol_forecast, record.aggregator_weights = self._aggregate(record, date)

        return record

    def _compute_influence_edges(self, record: DebateRecord) -> list:
        """Compute influence edges based on adjustment shifts between rounds."""
        edges = []
        for r in range(1, self.n_rounds):
            for target_id in AGENT_IDS:
                target_prev = record.get_opinion(target_id, r)
                target_curr = record.get_opinion(target_id, r + 1)
                if target_prev is None or target_curr is None:
                    continue

                total_shift = abs(target_curr.vol_adjustment - target_prev.vol_adjustment)
                if total_shift < 0.002:
                    continue

                for source_id in AGENT_IDS:
                    if source_id == target_id:
                        continue
                    source_op = record.get_opinion(source_id, r)
                    if source_op is None:
                        continue

                    shift_dir = target_curr.vol_adjustment - target_prev.vol_adjustment
                    gap_to_source = source_op.vol_adjustment - target_prev.vol_adjustment

                    if abs(gap_to_source) < 0.002:
                        continue
                    if np.sign(shift_dir) != np.sign(gap_to_source):
                        continue

                    weight = min(abs(shift_dir) / abs(gap_to_source), 1.0)
                    if weight > 0.05:
                        edges.append({
                            "source": source_id, "source_round": r,
                            "target": target_id, "target_round": r + 1,
                            "weight": round(weight, 4),
                        })
        return edges

    def _aggregate(self, record: DebateRecord, date: str) -> tuple:
        """Aggregate adjustments into final vol forecast.
        final_vol = persistence_vol + confidence-weighted mean adjustment.
        No LLM call needed - pure math aggregation for cleaner attribution.
        """
        persistence = getattr(self, '_persistence_vol', 0.25)
        final_opinions = record.get_round_opinions(self.n_rounds)
        if not final_opinions:
            return persistence, {}

        # Confidence-weighted mean adjustment
        total_w = sum(max(op.confidence, 0.01) for op in final_opinions)
        weighted_adj = sum(
            op.vol_adjustment * max(op.confidence, 0.01)
            for op in final_opinions
        ) / total_w

        final_vol = max(persistence + weighted_adj, 0.05)  # floor at 5%

        weights = {op.agent_id: max(op.confidence, 0.01) / total_w for op in final_opinions}
        return final_vol, weights
        return avg, {}

    def get_stats(self) -> dict:
        return {
            "total_llm_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "failed_calls": self.fail_count,
            "avg_tokens_per_call": self.total_tokens / max(self.call_count, 1),
            "success_rate": self.call_count / max(self.call_count + self.fail_count, 1),
        }


# ── Influence Graph ───────────────────────────────────────

class InfluenceGraph:
    """Debate influence graph for Myerson attribution."""

    def __init__(self, record: DebateRecord):
        self.record = record
        self.edges = record.influence_edges

    def get_connected_components(self, agent_subset: set) -> list:
        """Find connected components among agent subset (undirected)."""
        if not agent_subset:
            return []
        undirected = {a: set() for a in agent_subset}
        for edge in self.edges:
            s, t = edge["source"], edge["target"]
            if s in agent_subset and t in agent_subset:
                undirected[s].add(t)
                undirected[t].add(s)

        visited = set()
        components = []
        for agent in agent_subset:
            if agent in visited:
                continue
            component = set()
            queue = [agent]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                component.add(cur)
                for nb in undirected.get(cur, set()):
                    if nb not in visited:
                        queue.append(nb)
            components.append(component)
        return components

    def get_agent_degree(self) -> dict:
        """Count edges per agent (in + out)."""
        degree = {a: 0 for a in AGENT_IDS}
        for edge in self.edges:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1
        return degree

    def to_dict(self) -> dict:
        return {"edges": self.edges, "n_agents": N_AGENTS}


# ── Attribution Engine ────────────────────────────────────

class AttributionEngine:
    """Shapley and Myerson attribution for volatility forecast errors."""

    def __init__(self, n_samples: int = 1000, seed: int = 42):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def _counterfactual_error(self, record: DebateRecord, agent_subset: set) -> float:
        """Compute squared forecast error for a coalition of agents.

        Agents in subset contribute their adjustment.
        Agents NOT in subset contribute 0 adjustment (= persistence).
        Final forecast = persistence + weighted mean of active adjustments.
        """
        max_round = max(r for (_, r) in record.opinions.keys())
        persistence = getattr(record, '_persistence_vol', 0.25)
        adjustments, weights = [], []

        for agent_id in AGENT_IDS:
            opinion = record.get_opinion(agent_id, max_round)
            if agent_id in agent_subset and opinion is not None:
                adjustments.append(opinion.vol_adjustment)
                weights.append(max(opinion.confidence, 0.01))
            else:
                adjustments.append(0.0)  # no adjustment = persistence
                weights.append(0.01)

        total_w = sum(weights)
        weighted_adj = sum(a * w for a, w in zip(adjustments, weights)) / total_w
        pred = max(persistence + weighted_adj, 0.05)

        if np.isnan(record.actual_vol):
            return 0.0
        return (pred - record.actual_vol) ** 2

    def shapley_values(self, record: DebateRecord) -> dict:
        """Monte Carlo Shapley values for vol forecast error."""
        shapley = {a: 0.0 for a in AGENT_IDS}

        for _ in range(self.n_samples):
            perm = self.rng.permutation(AGENT_IDS).tolist()
            current = set()
            prev_err = self._counterfactual_error(record, current)

            for agent in perm:
                current.add(agent)
                curr_err = self._counterfactual_error(record, current)
                shapley[agent] += curr_err - prev_err
                prev_err = curr_err

        for a in AGENT_IDS:
            shapley[a] /= self.n_samples
        return shapley

    def myerson_values(self, record: DebateRecord, graph: InfluenceGraph) -> dict:
        """Monte Carlo Myerson values (graph-restricted Shapley)."""
        myerson = {a: 0.0 for a in AGENT_IDS}

        for _ in range(self.n_samples):
            perm = self.rng.permutation(AGENT_IDS).tolist()
            current = set()

            for agent in perm:
                new_set = current | {agent}
                components = graph.get_connected_components(new_set)

                # Find component containing the new agent
                agent_comp = {agent}
                for comp in components:
                    if agent in comp:
                        agent_comp = comp
                        break

                err_with = self._counterfactual_error(record, agent_comp)
                err_without = self._counterfactual_error(record, agent_comp - {agent})
                myerson[agent] += err_with - err_without
                current = new_set

        for a in AGENT_IDS:
            myerson[a] /= self.n_samples
        return myerson

    def classify_behaviors(self, record: DebateRecord) -> dict:
        """Classify each agent's debate behavior."""
        behaviors = {}
        max_round = max(r for (_, r) in record.opinions.keys())

        final_ops = record.get_round_opinions(max_round)
        majority_adj = np.median([op.vol_adjustment for op in final_ops]) if final_ops else 0.0

        for agent_id in AGENT_IDS:
            first = record.get_opinion(agent_id, 1)
            last = record.get_opinion(agent_id, max_round)
            if first is None or last is None:
                behaviors[agent_id] = "unknown"
                continue

            total_shift = abs(last.vol_adjustment - first.vol_adjustment)
            shift_toward_majority = (
                abs(majority_adj) > 0.001 and
                np.sign(last.vol_adjustment - first.vol_adjustment)
                == np.sign(majority_adj - first.vol_adjustment)
            )

            if total_shift < 0.002:
                behaviors[agent_id] = "anchored"
            elif last.confidence > 0.85 and total_shift > 0.005:
                behaviors[agent_id] = "overconfident"
            elif shift_toward_majority and total_shift > 0.003:
                behaviors[agent_id] = "herding"
            else:
                behaviors[agent_id] = "independent"

        return behaviors

    def attribution_report(self, record: DebateRecord, graph: InfluenceGraph) -> dict:
        """Full four-dimensional attribution report."""
        shapley = self.shapley_values(record)
        myerson = self.myerson_values(record, graph)
        behaviors = self.classify_behaviors(record)
        max_round = max(r for (_, r) in record.opinions.keys())

        # WHO: highest absolute attribution
        who = max(shapley, key=lambda k: abs(shapley[k]))

        # WHEN: round with largest aggregate shift
        when = 1
        max_shift = 0
        for r in range(2, max_round + 1):
            shift = sum(
                abs(record.get_opinion(a, r).vol_adjustment - record.get_opinion(a, r - 1).vol_adjustment)
                for a in AGENT_IDS
                if record.get_opinion(a, r) and record.get_opinion(a, r - 1)
            )
            if shift > max_shift:
                max_shift = shift
                when = r

        # WHAT: dominant behavior
        from collections import Counter
        what = Counter(behaviors.values()).most_common(1)[0][0] if behaviors else "unknown"

        # HOW: top influence path
        top_agent = max(myerson, key=lambda k: abs(myerson[k]))
        path = [top_agent]
        visited = {top_agent}
        current = top_agent
        for r in range(1, max_round):
            best_next, best_w = None, 0
            for edge in graph.edges:
                if edge["source"] == current and edge["source_round"] == r:
                    if edge["target"] not in visited and edge["weight"] > best_w:
                        best_next, best_w = edge["target"], edge["weight"]
            if best_next:
                path.append(best_next)
                visited.add(best_next)
                current = best_next

        # Shapley vs Myerson divergence
        divergence = {
            a: abs(shapley[a] - myerson[a]) for a in AGENT_IDS
        }

        forecast_error = (record.final_vol_forecast - record.actual_vol) ** 2 \
            if not np.isnan(record.actual_vol) else None

        return {
            "date": record.date,
            "forecast_error": forecast_error,
            "vol_forecast": record.final_vol_forecast,
            "actual_vol": record.actual_vol,
            "shapley": shapley,
            "myerson": myerson,
            "shapley_myerson_divergence": divergence,
            "behaviors": behaviors,
            "who": who,
            "when": when,
            "what": what,
            "how": path,
            "influence_edges": len(graph.edges),
            "graph_degree": graph.get_agent_degree(),
        }


# ── Data Preparation Helpers ─────────────────────────────

def prepare_market_data(row: pd.Series, horizon: int = 20) -> str:
    """Common market snapshot for all agents."""
    vol_col = f"wti_vol_{horizon}d"
    parts = []
    parts.append(f"WTI Close: ${row.get('wti_price', 0):.2f}")
    if pd.notna(row.get('wti_return')):
        parts.append(f"Daily Return: {row['wti_return'] * 100:.2f}%")
    if pd.notna(row.get(vol_col)):
        parts.append(f"{horizon}d Realized Vol: {row[vol_col]:.4f}")
    if pd.notna(row.get('wti_vol_60d')):
        parts.append(f"60d Realized Vol: {row['wti_vol_60d']:.4f}")
    return " | ".join(parts)


def prepare_agent_data(row: pd.Series, gdelt_row: pd.Series = None, horizon: int = 20) -> dict:
    """Prepare specialist data for each agent. Each agent sees ONLY its domain data."""
    data = {}
    vol_col = f"wti_vol_{horizon}d"

    # Geopolitical: GDELT conflict data only
    geo_parts = []
    if gdelt_row is not None:
        geo_parts.append(f"Oil-region events today: {gdelt_row.get('n_events_oil', 0):.0f}")
        geo_parts.append(f"Conflict events: {gdelt_row.get('oil_n_conflict_events', 0):.0f}")
        geo_parts.append(f"Conflict share of all events: {gdelt_row.get('oil_conflict_share', 0):.1%}")
        geo_parts.append(f"Material conflict share: {gdelt_row.get('oil_material_conflict_share', 0):.1%}")
        geo_parts.append(f"Cooperation-conflict balance (Goldstein): {gdelt_row.get('oil_goldstein_mean', 0):.2f}")
        geo_parts.append(f"Verbal cooperation events: {gdelt_row.get('oil_verbal_coop', 0):.0f}")
        geo_parts.append(f"Material conflict events: {gdelt_row.get('oil_material_conflict', 0):.0f}")
        geo_parts.append(f"Net cooperation index: {gdelt_row.get('oil_net_coop', 0):.3f}")
    data["geopolitical"] = "\n".join(geo_parts) if geo_parts else "No conflict data available today."

    # Macro demand: Chinese equity (demand proxy) + momentum + price level
    macro_parts = []
    if pd.notna(row.get('SHANGHAI_COMPOSITE')):
        macro_parts.append(f"Shanghai Composite index: {row['SHANGHAI_COMPOSITE']:.1f}")
    if pd.notna(row.get('wti_mom_20d')):
        macro_parts.append(f"Oil price 20-day momentum: {row['wti_mom_20d']:.4f}")
    if pd.notna(row.get('wti_mom_5d')):
        macro_parts.append(f"Oil price 5-day momentum: {row['wti_mom_5d']:.4f}")
    if pd.notna(row.get('wti_price')):
        macro_parts.append(f"Current WTI price: ${row['wti_price']:.2f}")
    data["macro_demand"] = "\n".join(macro_parts) if macro_parts else "No macro data available."

    # Monetary: rates, yields, DXY (no oil price or vol)
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
    v20 = row.get(vol_col, None)
    v60 = row.get('wti_vol_60d', None)
    if pd.notna(v20) and pd.notna(v60) and v60 > 0:
        tech_parts.append(f"Short/long vol ratio (20d/60d): {v20/v60:.2f}")
    if pd.notna(row.get('wti_return')):
        tech_parts.append(f"Latest daily return: {row['wti_return'] * 100:+.2f}%")
    data["technical"] = "\n".join(tech_parts) if tech_parts else "No technical data available."

    # Sentiment: GDELT tone/volume only (no conflict breakdown)
    sent_parts = []
    if gdelt_row is not None:
        sent_parts.append(f"Oil news volume: {gdelt_row.get('oil_mentions_sum', 0):.0f} mentions")
        sent_parts.append(f"Oil news article count: {gdelt_row.get('oil_articles_sum', 0):.0f}")
        sent_parts.append(f"Oil media tone (positive=cooperative): {gdelt_row.get('oil_tone_mean', 0):.2f}")
        sent_parts.append(f"Tone volatility: {gdelt_row.get('oil_tone_std', 0):.2f}")
        sent_parts.append(f"Global media tone: {gdelt_row.get('global_tone_mean', 0):.2f}")
    data["sentiment"] = "\n".join(sent_parts) if sent_parts else "No sentiment data available."

    # Cross-market: VIX + yield spread (no oil-specific data)
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


# ── Single-Agent Baseline ─────────────────────────────────

class SingleAgentBaseline:
    """Single LLM agent with ALL information (no debate). For ablation."""

    def __init__(self, model: str = MODEL, temperature: float = 0.3, call_delay: float = 0.0):
        self.client = get_client()
        self.model = model
        self.temperature = temperature
        self.call_delay = call_delay
        self.call_count = 0
        self.total_tokens = 0
        self.fail_count = 0

    def forecast(self, date: str, market_data: str, agent_data: dict,
                 persistence_vol: float = 0.25) -> float:
        """Single LLM call with all agent data combined."""
        prompt = SINGLE_AGENT_TEMPLATE.format(
            date=date, market_data=market_data,
            persistence_vol=persistence_vol,
            geo_data=agent_data.get("geopolitical", "N/A"),
            macro_data=agent_data.get("macro_demand", "N/A"),
            monetary_data=agent_data.get("monetary", "N/A"),
            supply_data=agent_data.get("supply_opec", "N/A"),
            technical_data=agent_data.get("technical", "N/A"),
            sentiment_data=agent_data.get("sentiment", "N/A"),
            cross_market_data=agent_data.get("cross_market", "N/A"),
        )
        for attempt in range(8):
            try:
                time.sleep(self.call_delay)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a senior oil market risk analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                self.call_count += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    raw = raw.rsplit("```", 1)[0]
                data = json.loads(raw.strip())
                return float(data.get("vol_forecast", persistence_vol))
            except Exception as e:
                wait = min(2 ** attempt * 2, 60)
                if attempt < 7:
                    time.sleep(wait)
                else:
                    return persistence_vol


# ── Intervention Experiment ───────────────────────────────

class InterventionExperiment:
    """Test whether attribution-targeted intervention reduces forecast error."""

    def __init__(self, attrib_engine: AttributionEngine):
        self.attrib_engine = attrib_engine

    def run_intervention(self, record: DebateRecord, graph: InfluenceGraph,
                         k: int = 2) -> dict:
        """Compare intervention strategies.

        For each strategy, replace top-k agents' opinions with neutral baseline,
        re-aggregate, and measure new error.

        Args:
            record: original debate record
            graph: influence graph
            k: number of agents to intervene on

        Returns:
            dict with error under each strategy
        """
        actual = record.actual_vol
        if np.isnan(actual):
            return {}

        original_error = (record.final_vol_forecast - actual) ** 2
        shapley = self.attrib_engine.shapley_values(record)
        myerson = self.attrib_engine.myerson_values(record, graph)

        # Rank agents by absolute attribution
        shapley_ranked = sorted(AGENT_IDS, key=lambda a: abs(shapley[a]), reverse=True)
        myerson_ranked = sorted(AGENT_IDS, key=lambda a: abs(myerson[a]), reverse=True)

        # Random baseline (average over multiple random selections)
        rng = np.random.RandomState(42)
        random_errors = []
        for _ in range(20):
            random_agents = set(rng.choice(AGENT_IDS, size=k, replace=False))
            err = self._intervene_and_predict(record, random_agents, actual)
            random_errors.append(err)
        random_error = np.mean(random_errors)

        # Shapley-targeted intervention
        shapley_targets = set(shapley_ranked[:k])
        shapley_error = self._intervene_and_predict(record, shapley_targets, actual)

        # Myerson-targeted intervention
        myerson_targets = set(myerson_ranked[:k])
        myerson_error = self._intervene_and_predict(record, myerson_targets, actual)

        # No intervention
        no_intervene_error = original_error

        return {
            "original_error": original_error,
            "no_intervention": no_intervene_error,
            "random_intervention": random_error,
            "shapley_intervention": shapley_error,
            "myerson_intervention": myerson_error,
            "shapley_targets": list(shapley_targets),
            "myerson_targets": list(myerson_targets),
            "shapley_reduction": (original_error - shapley_error) / max(original_error, 1e-10),
            "myerson_reduction": (original_error - myerson_error) / max(original_error, 1e-10),
            "random_reduction": (original_error - random_error) / max(original_error, 1e-10),
        }

    def _intervene_and_predict(self, record: DebateRecord,
                                agents_to_replace: set, actual: float) -> float:
        """Replace specified agents with 0 adjustment (persistence) and re-aggregate."""
        max_round = max(r for (_, r) in record.opinions.keys())
        persistence = getattr(record, '_persistence_vol', 0.25)
        adjustments, weights = [], []

        for agent_id in AGENT_IDS:
            opinion = record.get_opinion(agent_id, max_round)
            if agent_id in agents_to_replace:
                adjustments.append(0.0)  # no adjustment = persistence
                weights.append(0.01)
            elif opinion is not None:
                adjustments.append(opinion.vol_adjustment)
                weights.append(max(opinion.confidence, 0.01))
            else:
                adjustments.append(0.0)
                weights.append(0.01)

        total_w = sum(weights)
        weighted_adj = sum(a * w for a, w in zip(adjustments, weights)) / total_w
        pred = max(persistence + weighted_adj, 0.05)
        return (pred - actual) ** 2


# ── Quick Test ────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing 7-Agent Volatility Debate System")
    print("=" * 60)

    engine = DebateEngine(n_rounds=2, temperature=0.3)  # 2 rounds for speed

    # Crisis scenario: high-vol day
    record = engine.run_debate(
        date="2022-03-07",
        market_data="WTI: $119.40 (+7.1%) | 20d Vol: 0.58 | 60d Vol: 0.42",
        agent_data={
            "geopolitical": "Oil-country events: 28400\nConflict events: 12300\nConflict share: 0.433\nGoldstein: -2.41\nContext: Russia-Ukraine war, Western sanctions on Russian oil.",
            "macro_demand": "Shanghai Composite: declining. China lockdowns ongoing.\n20d momentum: +0.18",
            "monetary": "FED_FUNDS: 0.25\nYIELD_10Y: 1.78\nDXY: 98.5\nDXY daily: +0.3%\nFed expected to hike.",
            "supply_opec": "Russia produces ~10M bpd. Sanctions threaten 3-5M bpd supply.\nOPEC declining to boost output.\n5d momentum: +0.22",
            "technical": "20d vol: 0.58 (crisis level)\n60d vol: 0.42\nLatest return: +7.1%\nVol expanding rapidly. Prior day vol: 0.51.",
            "sentiment": "Oil news volume: 185000 mentions (5x normal)\nMedia tone: -3.2 (very negative)\nGlobal tone: -1.8\nWar coverage dominating.",
            "cross_market": "VIX: 36.4\nVIX change: +4.2\nEquity selloff. Flight to gold.\nCredit spreads widening.",
        },
        actual_vol=0.62,  # next day realized vol
    )

    print(f"\nDebate completed:")
    print(f"  Opinions: {len(record.opinions)} ({N_AGENTS} agents x {engine.n_rounds} rounds)")
    print(f"  Influence edges: {len(record.influence_edges)}")
    print(f"  Final vol forecast: {record.final_vol_forecast:.4f}")
    print(f"  Actual vol: {record.actual_vol:.4f}")
    print(f"  Aggregator weights: {record.aggregator_weights}")

    # Round-by-round summary
    for r in range(1, engine.n_rounds + 1):
        ops = record.get_round_opinions(r)
        vols = [f"{op.agent_id}: {op.vol_forecast:.3f}" for op in ops]
        print(f"\n  Round {r}: {', '.join(vols)}")

    # Attribution
    print("\n" + "=" * 60)
    print("Running attribution (Shapley + Myerson)...")
    graph = InfluenceGraph(record)
    attrib = AttributionEngine(n_samples=500)
    report = attrib.attribution_report(record, graph)

    print(f"\n--- Four-Dimensional Attribution ---")
    print(f"  WHO: {report['who']} (most responsible agent)")
    print(f"  WHEN: Round {report['when']} (critical shift round)")
    print(f"  WHAT: {report['what']} (dominant behavior)")
    print(f"  HOW: {' -> '.join(report['how'])} (influence path)")

    print(f"\n--- Shapley Values ---")
    for a in sorted(report['shapley'], key=lambda x: abs(report['shapley'][x]), reverse=True):
        print(f"  {a:20s}: {report['shapley'][a]:+.6f}")

    print(f"\n--- Myerson Values ---")
    for a in sorted(report['myerson'], key=lambda x: abs(report['myerson'][x]), reverse=True):
        print(f"  {a:20s}: {report['myerson'][a]:+.6f}")

    print(f"\n--- Shapley vs Myerson Divergence ---")
    for a in sorted(report['shapley_myerson_divergence'],
                     key=lambda x: report['shapley_myerson_divergence'][x], reverse=True):
        print(f"  {a:20s}: {report['shapley_myerson_divergence'][a]:.6f}")

    print(f"\n--- Behaviors ---")
    for a, b in report['behaviors'].items():
        print(f"  {a:20s}: {b}")

    print(f"\n--- Graph ---")
    print(f"  Edges: {report['influence_edges']}")
    print(f"  Degree: {report['graph_degree']}")

    print(f"\nLLM stats: {engine.get_stats()}")
