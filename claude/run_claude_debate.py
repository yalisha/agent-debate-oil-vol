#!/usr/bin/env /opt/miniconda3/bin/python
"""
Orchestrate multi-agent debate using Claude Code sub-agents.
Reads prepared data, outputs agent prompts for batch execution,
and aggregates results.

This script is designed to be called by Claude Code, not run standalone.
It generates the prompts and parses results.
"""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
AGENTS = ['geopolitical', 'macro', 'monetary', 'supply', 'technical', 'sentiment', 'cross_market']

AGENT_ROLES = {
    'geopolitical': 'GEOPOLITICAL RISK ANALYST specializing in oil markets',
    'macro': 'MACRO DEMAND ANALYST specializing in oil markets',
    'monetary': 'MONETARY POLICY ANALYST specializing in oil markets',
    'supply': 'SUPPLY/OPEC ANALYST specializing in oil markets',
    'technical': 'TECHNICAL/QUANTITATIVE ANALYST specializing in oil volatility',
    'sentiment': 'SENTIMENT/NEWS ANALYST specializing in oil markets',
    'cross_market': 'CROSS-MARKET ANALYST specializing in inter-market dynamics',
}


def make_specialist_data(day, agent):
    if agent == 'geopolitical':
        return (f"Oil-country events: {day['geo_events']}\n"
                f"Conflict events: {day['conflict_events']}\n"
                f"Conflict share: {day['conflict_share']}\n"
                f"Goldstein mean: {day['goldstein']}\nTone: {day['tone']}")
    elif agent == 'macro':
        return f"Shanghai Composite: {day['shanghai']}\n20d momentum: {day['mom_20d']}"
    elif agent == 'monetary':
        return (f"FED_FUNDS: {day['fed_funds']}%\nYIELD_2Y: {day['yield_2y']}%, "
                f"YIELD_10Y: {day['yield_10y']}%\nDXY: {day['dxy']}, "
                f"change: {day['dxy_change']}%\nSpread: {day['spread']}")
    elif agent == 'supply':
        return (f"WTI price: ${day['wti_price']}\n"
                f"5d momentum: {day['mom_5d']}\n20d momentum: {day['mom_20d']}")
    elif agent == 'technical':
        return (f"20d vol: {day['vol_20d']}\n60d vol: {day['vol_60d']}\n"
                f"Daily return: {day['wti_return']}%")
    elif agent == 'sentiment':
        return (f"Oil mentions: {day['oil_mentions']}\n"
                f"Media tone: {day['tone']}\nNet cooperation: {day['net_coop']}")
    elif agent == 'cross_market':
        return (f"VIX: {day['vix']}\nVIX change: {day['vix_change']}\n"
                f"Spread change: {day['spread_change']}")


def make_r1_prompt(day, agent):
    role = AGENT_ROLES[agent]
    data = make_specialist_data(day, agent)
    return (
        f"You are a {role}.\n\n"
        f"Date: {day['date']}\n"
        f"Market: WTI ${day['wti_price']} | Return {day['wti_return']}% | "
        f"20d Vol: {day['vol_20d']} | 60d Vol: {day['vol_60d']}\n\n"
        f"Your data:\n{data}\n\n"
        f"Persistence baseline: {day['vol_20d']}. "
        f"Predict vol ADJUSTMENT for next 20 trading days.\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{\"vol_adjustment\": <float -0.5 to +0.5>, \"confidence\": <float 0-1>, '
        f'\"evidence\": \"<one sentence>\"}}'
    )


def make_r2_prompt(day, agent, r1_results):
    role = AGENT_ROLES[agent]
    own = r1_results[agent]
    others = "\n".join(
        f"- {a.title()}: {r1_results[a]['vol_adjustment']:+.2f}, "
        f"conf={r1_results[a]['confidence']}"
        for a in AGENTS if a != agent
    )
    return (
        f"You are a {role}. Round 2: revise after seeing others.\n\n"
        f"Date: {day['date']} | WTI ${day['wti_price']} | 20d Vol: {day['vol_20d']}\n\n"
        f"YOUR Round 1: adj={own['vol_adjustment']:+.2f}, conf={own['confidence']}\n\n"
        f"Others:\n{others}\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{\"vol_adjustment\": <float>, \"confidence\": <float 0-1>, '
        f'\"evidence\": \"<one sentence>\"}}'
    )


def aggregate(persistence, r2_results):
    adjs = [r['vol_adjustment'] for r in r2_results.values()]
    confs = [r['confidence'] for r in r2_results.values()]
    weighted = sum(a * c for a, c in zip(adjs, confs)) / sum(confs)
    return max(persistence + weighted, 0.01)


def classify_behavior(r1, r2):
    """Classify agent behavior between rounds."""
    shift = r2['vol_adjustment'] - r1['vol_adjustment']
    if abs(shift) < 0.005:
        return 'anchored'
    elif r2['confidence'] > 0.85 and abs(shift) > 0.02:
        return 'overconfident'
    # Check if moved toward median
    return 'herding' if abs(shift) > 0.005 else 'independent'


def parse_json_response(text):
    """Parse JSON from agent response, handling markdown wrapping."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        text = text.rsplit('```', 1)[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {"vol_adjustment": 0.0, "confidence": 0.5, "evidence": "parse error"}


if __name__ == '__main__':
    # Load data
    with open(BASE / 'feb2022_data.json') as f:
        days = json.load(f)
    print(f"Loaded {len(days)} trading days")
    for d in days:
        print(f"  {d['date']}: vol={d['vol_20d']}, target={d['fwd_rv_20d']}")
