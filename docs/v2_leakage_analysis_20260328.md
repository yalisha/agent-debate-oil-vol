# V2 Shapley Leakage Analysis Report

Date: 2026-03-28

## Background

During v2 restructuring (aligning with IJF reference paper Zhang et al. 2025),
ChatGPT cross-review flagged the Shapley look-ahead problem as the top priority
issue for reviewer defence. This triggered a systematic investigation.

## The Problem

In v1, Shapley values are node features fed into the GAT at prediction time:

```
node_feats[:, i, 0] = shapley_agent(t)      # LEAK
node_feats[:, i, 1] = myerson_agent(t)       # LEAK
node_feats[:, i, 4] = shapley_ma5_agent(t)   # LEAK (rolling of leaking values)
node_feats[:, i, 5] = shapley_std5_agent(t)  # LEAK (rolling of leaking values)
```

The leak mechanism:
- $\phi_i(t)$ is computed from the naive aggregation error using actual_vol(t)
- actual_vol(t) = std(returns t+1 to t+20) * sqrt(252)
- At prediction time t, returns t+1 to t+20 are NOT yet observed
- The model effectively receives a signal about the future target

Leak-free features (behaviour encoding, degree, herding streak) use only
R1/R2 adjustment trajectories, which are fully determined at time t.

## Experiments

### Experiment 1: Lagged Shapley (lag=20)

Rationale: by day t, actual_vol(t-20) is fully observed, so $\phi_i(t-20)$
is leak-free.

Results:
```
DropEdge GAT (lagged):  RMSE=0.1521, DM_HAC=-0.405 (p=0.686)  n.s.
Learned GAT (lagged):   RMSE=0.1517, DM_HAC=-0.456 (p=0.649)  n.s.
MLP (lagged):           RMSE=0.1480, DM_HAC=-0.839 (p=0.402)  n.s.
GAT vs MLP:             DM_HAC=0.588 (p=0.557)  n.s.
```

Finding: lagged Shapley carries no useful signal. RMSE matches the v1
No-Shapley baseline (0.1520). Information from 20 days ago is stale.

### Experiment 2: Enhanced Leak-Free Features (4 variants)

Tested four feature sets, each with DropEdge GAT and MLP:

| Variant | Node dim | Features | GAT RMSE | MLP RMSE | GAT vs HAR |
|---------|----------|----------|----------|----------|------------|
| A | 7 | Lagged Shapley + augmented context | 0.1513 | 0.1523 | p=0.624 |
| B | 8 | Rich behaviour (no Shapley) | 0.1715 | 0.1566 | p=0.117 |
| C | 11 | Lagged Shapley + rich combined | 0.1688 | 0.1596 | p=0.217 |
| D | 3 | Minimal (beh/degree/streak only) | 0.1505 | 0.1509 | p=0.574 |

Rich behaviour features (Variant B) included:
- Rolling 10-day herding frequency per agent
- Behaviour diversity (distinct types in last 10 days)
- Independent indicator
- Herding-independent balance
- Rolling degree

Augmented context (all variants) added:
- Rolling debate accuracy (lag 20 days)
- Herding trend (5-day MA)
- Debate-single gap

GAT vs MLP comparisons (no variant shows graph advantage):
```
Variant A: GAT=0.1513 vs MLP=0.1523, DM_HAC=-0.143 (p=0.886)
Variant B: GAT=0.1715 vs MLP=0.1566, DM_HAC= 1.586 (p=0.113)  MLP better
Variant C: GAT=0.1688 vs MLP=0.1596, DM_HAC= 1.016 (p=0.310)  MLP better
Variant D: GAT=0.1505 vs MLP=0.1509, DM_HAC=-0.061 (p=0.952)
```

### Reference Points

```
v1 DropEdge GAT (concurrent Shapley): RMSE=0.1171, DM=-3.212, p=0.0013 ***
v1 No-Shapley baseline:               RMSE=0.1520, DM=-0.420, p=0.675  n.s.
HAR benchmark:                         RMSE=0.1549
Persistence:                           RMSE=0.1551
Naive debate:                          RMSE=0.1512, DM=-1.494, p=0.135  n.s.
GARCH(1,1):                            RMSE=0.1366, DM=-2.604, p=0.009  **
```

## Key Findings

### 1. The entire v1 accuracy gain was driven by Shapley look-ahead

v1 DropEdge GAT (0.1171) vs best leak-free (0.1505): the 22% RMSE improvement
was attributable to the model receiving future information through Shapley
features. Without them, GAT performance drops to the HAR/persistence level.

### 2. Graph structure provides no benefit without look-ahead

In all leak-free variants, GAT and MLP perform equivalently (or MLP is better).
The graph structure cannot differentiate agent quality without an oracle signal
about which agents are currently accurate.

Why: the only agent-differentiating leak-free features available are behavioural
classifications (herding/anchoring/independent/overconfident), debate graph
degree, and herding streaks. These describe HOW agents behave but not WHETHER
their predictions are accurate. Accuracy assessment requires comparing to the
actual outcome, which is future information.

### 3. More features hurt rather than help

Variants B (8-dim) and C (11-dim) performed worse than D (3-dim), especially
for GAT. With only 7 nodes and limited training data per walk-forward window,
additional features introduce noise and overfitting risk.

### 4. The naive debate forecast is the effective ceiling

The best leak-free models (RMSE ~0.1505-0.1513) perform similarly to the naive
debate forecast (0.1512). Without look-ahead, the learned combination cannot
extract signal beyond what confidence-weighted averaging already captures.

### 5. GARCH remains the only significant non-leaking model

Among all 16+ methods tested, only GARCH(1,1) significantly outperforms HAR
after HAC correction (RMSE=0.1366, p=0.009), using a purely statistical
model with no LLM input at all.

## Root Cause Analysis

The fundamental issue is an information asymmetry:

- To weight agents optimally, the combination mechanism needs to know which
  agents are currently providing good forecasts
- Knowing which agent is "good" requires comparing its prediction to the
  actual outcome
- The actual outcome is not available until 20 days in the future
- Any feature that encodes this comparison (Shapley, Myerson, or any
  accuracy-based metric) inherits the look-ahead

The available leak-free agent features (behaviour, degree, herding) describe
agent interaction dynamics but not agent forecast quality. The GAT can learn
patterns in agent interactions, but without a quality signal, it cannot
determine which interaction patterns produce better forecasts.

## Missing Data: Per-Agent Adjustments and Confidence

The current debate CSV does not store per-agent R1/R2 adjustments or
confidence scores. These would be the richest leak-free agent-level features:

- agent_i_R1_adjustment: what agent i actually predicted (available at t)
- agent_i_R2_adjustment: revised prediction after debate (available at t)
- agent_i_confidence: self-reported confidence in [0,1] (available at t)
- agent_i_revision_magnitude: |R2 - R1| (available at t)

These features describe WHAT each agent predicts and HOW CERTAIN it is,
not just WHETHER it herded. This is substantially more informative.

Re-running the debate to save these values would require ~1285 API calls
(one per trading day) to the Gemini 3 Flash endpoint.

## Implications for the Paper

### What still holds
- The SDD debate protocol generates diverse base forecasts from LLM agents
- Multi-agent herding behaviour is observable and quantifiable (43.4% rate)
- HAC correction is essential for overlapping-horizon evaluation (reverses
  conclusions for multiple baselines)
- Ex-post Shapley/Myerson attribution decomposes agent contribution
  meaningfully (as a diagnostic, not a prediction feature)
- GARCH(1,1) significantly outperforms HAR for oil volatility (p=0.009)

### What does not hold
- "Learned graph-based combination is the source of forecasting gains" -
  the gain was from look-ahead leakage
- "GAT significantly outperforms MLP" - graph structure adds no value
  without agent quality signals
- "The framework achieves RMSE=0.1171" - this number is contaminated

### Options for the paper

**Option A: Re-run debate with per-agent features saved**
- Adds R1/R2 adjustments and confidence as leak-free node features
- Most promising path to recover a genuine accuracy result
- Cost: ~1285 API calls
- Risk: no guarantee these features enable significant improvement

**Option B: Reposition the paper**
- Core contribution shifts from accuracy to methodology and interpretability
- HAC correction for overlapping horizons as a methodological contribution
- Multi-agent LLM behaviour analysis as a substantive contribution
- Ex-post Shapley attribution as an interpretability tool
- Honest reporting: "leak-free combination remains an open problem"
- Risk: weaker paper without a clear accuracy result

**Option C: Hybrid approach**
- Re-run debate (Option A) for richer features
- If significant result obtained: accuracy + interpretability paper
- If not: fall back to Option B framing
- Highest expected value but requires the experiment investment

## Files

- `src/optimized_gat_v2.py`: experiment script (4 feature variants)
- `results/optimized_gat_v2_lagged_shapley.csv`: initial lagged Shapley results
- `results/optimized_gat_v2_feature_experiments.csv`: full 4-variant results
- `manuscript/v1/`: archived v1 manuscript (contaminated numbers)
- `manuscript/v2/outline_v2.md`: v2 outline (needs revision based on findings)
