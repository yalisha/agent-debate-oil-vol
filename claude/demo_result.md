# Claude Code Agent Debate Demo

## Setup
- Date: 2022-02-24 (Russia invades Ukraine)
- Backend: Claude Opus 4.6 via Claude Code sub-agents
- 7 specialist agents, 2 rounds of debate
- Target: forward-looking 20d realized volatility

## Round 1 (Independent Forecasts)

| Agent | vol_adjustment | confidence | evidence |
|-------|---------------|------------|----------|
| Geopolitical | +0.12 | 0.85 | Invasion drives substantial upward pressure |
| Macro | +0.18 | 0.85 | Extreme geopolitical risk premium, supply disruption |
| Monetary | +0.15 | 0.82 | DXY decline, tightening cycle amplifies vol |
| Supply/OPEC | +0.18 | 0.85 | Extreme supply disruption, tight spare capacity |
| Technical | +0.12 | 0.72 | Inverted vol term structure signals reversion |
| Sentiment | +0.18 | 0.92 | Record news volume, deeply negative tone |
| Cross-Market | +0.12 | 0.72 | Elevated VIX, cross-market fear spillover |

## Round 2 (After Seeing Others)

| Agent | R1 | R2 | Shift | Behavior |
|-------|-----|-----|-------|----------|
| Geopolitical | +0.12 | +0.16 | +0.04 | herding |
| Macro | +0.18 | +0.17 | -0.01 | independent |
| Monetary | +0.15 | +0.16 | +0.01 | herding |
| Supply/OPEC | +0.18 | +0.17 | -0.01 | independent |
| Technical | +0.12 | +0.15 | +0.03 | herding |
| Sentiment | +0.18 | +0.17 | -0.01 | independent |
| Cross-Market | +0.12 | +0.16 | +0.04 | herding |

## Results

- Persistence baseline: 0.3138
- Debate prediction: 0.4769
- Actual fwd 20d RV: 0.8752
- Debate error: 0.3983
- Persistence error: 0.5614
- Improvement vs persistence: 29.1%

## Behavioral Analysis

- R1 std: 0.028, R2 std: 0.007 (convergence: 75%)
- Classic herding pattern: low-confidence agents moved most toward consensus
- Information cascade: Technical and Cross-Market (conf 0.72) shifted +0.03/+0.04
- High-confidence Sentiment (0.92) barely moved (-0.01)
