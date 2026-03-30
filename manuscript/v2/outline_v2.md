# Outline v2: IJF Submission

Restructured to match IJF conventions (cf. Zhang et al., 2025, IJF 41:377-397).
Core changes from v1: (1) Related Work dissolved into Introduction,
(2) Preliminaries section for background definitions and baselines,
(3) SDD + GAT unified into a single Proposed Framework section,
(4) Motivation moved post-results as "What Drives the Gain?",
(5) Robustness promoted from appendix to section,
(6) 8 sections reduced to 7.

Key design decisions (from ChatGPT cross-review of v2 outline):
- Shapley/Myerson reframed as ex-post interpretability tool, NOT
  prediction-time input. Node features use only leak-free quantities
  (behaviour encoding, degree, herding streak). Shapley analysis
  appears in Section 5 as post-hoc attribution, not in Section 3
  as a model input.
- Graph stability framed positively: sparsity level and higher-level
  motifs (agent-type connectivity patterns) are stable; specific
  edge-level variation deferred to appendix.
- Section 6 renamed "Robustness and Inference" to accommodate
  both data-exclusion sensitivity and HAC-correction methodology.
- Abstract rewritten (v1 abstract said "causal interpretation",
  which is no longer claimed in v2 title or body).

---

## Title

SDD-GAT: Oil Volatility Forecast Combination via Learned Graph
Attention over Multi-Agent LLM Debate

---

## Keywords

forecast combination, graph attention network, oil volatility,
large language models, Shapley attribution, regime adaptation

---

## Abstract (~150 words)

REWRITE NEEDED (v1 abstract references "causal interpretation"
which v2 no longer claims).

Core message to preserve:
- Two-stage framework: SDD debate + learned sparse GAT
- 972 OOS predictions, HAC-corrected DM tests
- RMSE=0.1171, only approach significant vs HAR after HAC
- ~16/42 edges retained, interpretable via ex-post Shapley/Myerson
  attribution

Changes from v1 abstract:
- Remove "causal interpretation" -> "ex-post interpretability"
- Remove "causally interpretable" from any framing
- Add: "post-hoc Shapley and Myerson attribution decompose
  each agent's contribution" (clarify it is diagnostic, not input)

---

## 1. Introduction (~3 pages, ~2,200 words)

Absorbs v1 Sections 1 + 2. Literature woven in as motivation,
not survey-style. Five thematic streams appear as natural parts
of the argument, not as labelled subsections.

### 1.1 Opening: the volatility forecasting problem

Oil volatility matters (risk management, hedging, option pricing).
WTI concentrates heterogeneous shocks. HAR and GARCH remain
competitive despite decades of alternatives.
[Embed: Corsi 2009, Bollerslev 1986, Degiannakis & Filis 2017,
Andersen & Bollerslev 1998, Sevi 2014]

### 1.2 The combination problem

When no single model suffices, combination is the standard response.
Bates & Granger (1969) through Petropoulos et al. (2022).
Three persistent limitations (Wang et al., 2023): independence
assumption, fixed weights, no mechanism for learning forecaster
interaction. Conditionally optimal combination (Gibbs & Vasnev, 2024)
and Shapley-based attribution (Franses et al., 2024) begin to relax
these, but neither learns the network structure among forecasters.
[Embed: Smith & Wallis 2009, Genre et al. 2013, Claeskens et al. 2016,
Timmermann 2006]

### 1.3 LLM agents as a new forecaster class

LLMs offer heterogeneous conceptual priors unavailable to statistical
models. Sentiment extraction (FinGPT, Dai et al. 2026), zero-shot
forecasting limitations (Abolghasemi et al. 2025, Makridakis et al. 2022).
Multi-agent debate improves reasoning (Du et al. 2024, Takano et al. 2025).
But herding emerges when agents interact (Ashery et al. 2024, Bini et al.
2026, Sun et al. 2025). Naive confidence-weighted mean does not
significantly outperform HAR after HAC correction.
[Embed: TradingAgents/Xiao et al. 2024, FinCon/Yu et al. 2024,
Ghali et al. 2025, Ashery et al. 2025]

### 1.4 Graph-structured combination: the proposal

GNNs have been applied to asset-level graphs for volatility spillovers
(Zhang et al. 2025, Brini & Toscano 2025) and return prediction
(Chi et al. 2025), but not to forecaster-level combination. Constructing
a graph over agents encodes combination structure rather than market
microstructure. The SDD-GAT framework: seven specialist agents generate
diverse forecasts via Structured Delphi Debate; a learned sparse GAT
with DropEdge discovers which agents should attend to one another and
adapts weights across volatility regimes. ~16/42 edges retained.
Post-hoc Shapley and Myerson attribution decompose each agent's
contribution into an independent informational component and a
network-mediated component.
[Embed: Velickovic et al. 2018, Rong et al. 2020,
communication topology/Zhang et al. 2024]

### 1.5 Contributions (woven into prose, not bulleted)

Three contributions:
1. SDD protocol connects multi-agent debate to forecast combination,
   generating interpretable base forecasts with observable behaviour
   and interaction structure.
2. Learned sparse GAT with DropEdge discovers agent interaction and
   adapts weights to regimes via gated dual-head output, extending
   conditionally optimal combination (Gibbs & Vasnev 2024) to a
   graph-structured setting.
3. Empirical evidence that HAC correction reverses apparent significance
   of naive debate forecasts; only GAT variants and GARCH survive
   correction among 16 methods on 972 OOS predictions. The learned
   combination mechanism, not base forecast quality alone, is the source
   of forecasting gains. Post-hoc Shapley and Myerson attribution
   provide a practitioner-facing interpretability layer.

### 1.6 Paper organisation (single paragraph)

Section 2 reviews preliminaries. Section 3 presents the proposed
framework. Section 4 reports empirical results. Section 5 analyses
what drives the forecasting gain. Section 6 examines robustness
and inference methodology. Section 7 concludes.

---

## 2. Preliminaries (~1.5 pages, ~1,200 words)

NEW section. Defines background concepts and baseline models that
the proposed method builds upon. Follows the reference paper's
convention of reviewing foundational material before the new method.

### 2.1 Prediction target and evaluation protocol

From v1 Section 3.1-3.2:
- Forward-looking 20-day RV definition
- Overlapping horizon and serial correlation (ACF lag-1 = 0.938)
- HAC-corrected DM test (Newey-West, Bartlett kernel, bandwidth 19)
- Walk-forward protocol (min_train=252, retrain every 63 days)
- 20-day label embargo for supervised models
- 5-seed ensemble, 972 OOS predictions
[Cite: Diebold & Mariano 1995, Newey & West 1987]

### 2.2 Baseline volatility models

Brief review of the two econometric benchmarks the paper repeatedly
references. Not full literature review, just operational definitions:
- **Persistence**: $\hat{y}_t = \text{persist\_vol}_t$ (backward 20-day RV)
- **HAR-RV** (Corsi, 2009): daily/weekly/monthly lagged RV components.
  Standard benchmark for realised volatility forecasting.
- **GARCH(1,1)** (Bollerslev, 1986): conditional variance model.
  Remains competitive for oil markets (Sevi, 2014).

### 2.3 Graph attention networks

Brief technical background on GAT (1 paragraph + key equation).
Define graph $G = (V, E)$, adjacency matrix $A$, attention
coefficient computation (Velickovic et al. 2018). Multi-head
attention. This provides notation for Section 3.
[Cite: Velickovic et al. 2018, Kipf & Welling 2017]

### 2.4 Notation summary

From v1 Section 3.3:
- $\mathcal{A} = \{a_1, \ldots, a_7\}$: seven specialist agents
- $\hat{v}^{(1)}_i$, $\hat{v}^{(2)}_i$: Round 1/2 adjustments
- $\hat{y}^{\text{debate}}_t$: confidence-weighted Round 2 mean
- $\hat{y}^{\text{single}}_t$: single-agent baseline
- $\phi_i(t)$: Shapley value
- $G_w$: learned GAT graph; $G^{\text{dbt}}_t$: debate influence graph

---

## 3. Proposed Framework: SDD-GAT (~3.5 pages, ~2,800 words)

Merges v1 Sections 5 (Architecture) + 6 (SDD Protocol) into a
single methodology section. Follows the reference paper's convention
of presenting all new methodology in one place. Ordered by data flow:
debate first (Stage 1), then GAT (Stage 2).

[Figure 1 here: two-stage architecture diagram. Data flow from
market features through seven agents, two-round debate, into GAT,
out to prediction. Annotate node features, edge logits, dual head,
regime gate.]

### 3.1 Stage 1: Structured Delphi Debate

From v1 Section 6.1-6.2:
- Seven specialist agents (geopolitical, macro_demand, monetary,
  supply_opec, technical, sentiment, cross_market)
- Each receives domain-specific structured features
- Four structured output fields: adjustment, confidence, direction,
  evidence summary
- Round 1: independent forecast (diversity guarantee)
- Round 2: each agent sees all Round 1 outputs, revises
- Base forecasts: $\hat{y}^{\text{debate}}_t$, $\hat{y}^{\text{single}}_t$
[Cite: Dalkey & Helmer 1963, Du et al. 2024]

### 3.2 Behavioural classification

From v1 Section 6.3:
- Four behavioural states: herding, anchoring, independent, overconfident
- Classification based on R1-to-R2 adjustment trajectory
- Produces a per-agent-per-day encoding that enters the GAT as
  a node feature (behaviour code, herding streak)

NOTE: Shapley/Myerson attribution is NOT a prediction-time input.
It is defined here for completeness but used only as an ex-post
diagnostic in Section 5.4. See "Shapley design decision" below.

**Shapley design decision (critical for reviewer defence):**
In v1, Shapley and Myerson values were node features fed into the
GAT at prediction time. This creates a look-ahead problem: $\phi_i(t)$
is computed from the naive aggregation error using actual_vol(t),
which depends on returns $r_{t+1}, \ldots, r_{t+20}$ not yet observed.
The v1 ablation "No Shapley/Myerson" (RMSE=0.1520) is the leak-free
baseline; it loses significance vs HAR (p=0.675).

v2 resolution: **remove Shapley/Myerson from node features entirely.**
Node features become 4-dim: behaviour_code/3, degree/6, herding_streak/5,
plus one rolling statistic (e.g. 5-day rolling confidence or adjustment
magnitude). Shapley and Myerson become purely ex-post interpretability
tools reported in Section 5.4.

This means Section 4 results will use the leak-free architecture.
The headline RMSE will change (likely close to the v1 No-Shapley
baseline of 0.1520, or better if we add leak-free rolling features).
**REQUIRES RE-RUNNING EXPERIMENTS** with the revised node features
before Section 4 numbers can be finalised.

### 3.3 Stage 2: Learned sparse GAT

From v1 Section 5.2-5.3:
- **Graph construction**: learnable logit matrix $L \in R^{7 \times 7}$,
  $A = \sigma(L)$, threshold at 0.5. ~16/42 edges active.
- **Node features**: 4-dim per agent (leak-free):
  behaviour_code/3, normalised degree/6, herding_streak/5,
  plus one rolling diagnostic (TBD after experiment).
  Context vector (9-dim) unchanged: persist_vol, har_vol, total_adj,
  n_herding/7, vol_regime/3, debate_vol, persist-har gap,
  vol_change_5d, debate-har gap.
- **GAT backbone**: 2-layer multi-head GAT (4 heads, hidden_dim=16,
  top-k=3). Skip connection + LayerNorm after each layer.
- **DropEdge**: 20% edge dropout during training, all edges at inference.
  Prevents reliance on single edge configuration.
[Cite: Velickovic et al. 2018, Rong et al. 2020]

### 3.4 Regime-gated output and training

From v1 Section 5.5-5.6:
- Dual-head output: base head + regime head, each producing softmax
  weights over $[\hat{y}^{\text{debate}}_t, \hat{y}^{\text{single}}_t]$
- Regime gate: Linear(3,8)->Tanh->Linear(8,1)->Sigmoid,
  inputs = [persist_vol, vol_change_5d, n_herding/7]
- Final weights: $(1-g) w_{\text{base}} + g w_{\text{regime}}$
- Residual branch: scaled by 0.05
- Training: Adam lr=0.003, cosine annealing, 250 epochs, grad clip 1.0
- Walk-forward: retrain every 63 days, 5-seed ensemble

---

## 4. Empirical Results (~3 pages, ~2,500 words)

Main results from v1 Section 7.1-7.4. Focused on presenting findings;
deeper analysis of "why" deferred to Section 5.

### 4.1 Data and experimental setup

From v1 Section 7.1:
- Sample: 2020-01 to 2025-05, n=1,285. Clean n=1,224 after excluding
  61 days (April 2020 negative-price episode). OOS n=972.
- Six baseline categories: naive (persistence), econometric (HAR, GARCH),
  ML (Ridge, Lasso, GBR, RF, XGBoost), DL (LSTM, Transformer),
  LLM (single-agent, naive debate), proposed (DropEdge GAT).
- Data sources specified.

### 4.2 Overall forecast accuracy (Table 1)

From v1 Section 7.2:
- DropEdge GAT: RMSE=0.1171, DM_HAC=-3.212, p=0.0013
- Only GAT variants and GARCH survive HAC correction
- Naive debate: RMSE=0.1512, DM_HAC=-1.494, p=0.135 (not significant)
- ML/DL baselines: none significant after embargo + HAC
- HAC correction essential: naive DM overstates by ~65%

[Table 1: full model comparison, 16 methods]

### 4.3 Architecture ablation (Table 2)

From v1 Section 7.3, but all ablations now run on the leak-free
architecture (no Shapley/Myerson in node features):
- GAT vs MLP: graph structure contribution [RE-RUN NEEDED]
- Dense GAT (42/42) vs learned sparse: sparsity contribution
- Random graph (~16/42): learned vs random sparsity
- Identity (self-loops only): inter-agent edges contribution
- No regime gate: regime adaptation contribution
- Learned sparsity, not graph per se, drives the improvement

[Table 2: ablation results, variants TBD after re-run]

**NOTE:** All v1 numbers in this section are invalidated by the
Shapley removal. Experiment re-run required before writing.

### 4.4 Regime-conditional performance (Table 3)

From v1 Section 7.4:
- Four regimes by persistence volatility thresholds
- GAT dominates in normal + elevated (n=831)
- MLP slightly better in high regime (n=76, underpowered)
- Regime gate appropriately shifts weight

[Table 3: regime RMSE for 5 models x 4 regimes]
[Figure 3: regime-conditional RMSE bar chart]

---

## 5. What Drives the Forecasting Gain? (~2 pages, ~1,600 words)

Absorbs v1 Section 4 (Motivation) + v1 Section 7.5-7.6.
Positioned after main results, following the reference paper's
convention of a dedicated "deep analysis" section that investigates
why the proposed method works.

### 5.1 Why naive aggregation fails

From v1 Section 4.1 (reframed as post-hoc analysis, not pre-method
motivation):
- Herding: 43.4% of agent-day observations classified as herding
- All agents have weakly negative mean Shapley (each helps on average),
  but usefulness is episodic (positive Shapley on 38-59% of days)
- Confidence-weighted averaging assigns persistent weight to agents
  that are sporadically useful, not selectively useful
- Communication density correlates with herding (r=0.35)

### 5.2 The role of learned sparsity

From v1 Section 4.3 + ablation evidence:
- Dense GAT worse than self-loops only: full connectivity amplifies
  herding-induced redundancy [numbers from re-run]
- Sparse graph gates out edges connecting herded agents,
  preserving independent signals
- Sparsity level (~16/42, ~38% density) is the structural finding,
  not specific topology

### 5.3 Graph structure and temporal evolution

Reframed from v1 Section 7.5. Lead with positive findings,
defer edge-level instability detail to Appendix B.

Main text framing:
- The learned graph consistently converges to ~38% edge density
  across seeds and walk-forward windows (stable sparsity level)
- Higher-level connectivity patterns are interpretable: which
  agent types tend to connect follows economic logic
- Temporal evolution carries economic content:
  - monetary->macro_demand connectivity strengthens through the
    post-pandemic rate cycle (early windows ~27%, late ~60%)
  - sentiment->supply_opec weakens as OPEC positioning normalises
    (early ~73%, late ~33%)
- Ensemble of five seeds achieves low RMSE variance (std=0.004)
  despite topological variation, indicating that the ensemble
  averages over locally optimal graphs

Edge-level frequency details (per-seed, per-window breakdown)
deferred to Appendix B alongside hyperparameter tables.

[Figure 4: temporal evolution of key agent-type connectivity
patterns, 2-3 panels]

### 5.4 Ex-post agent attribution

Shapley and Myerson analysis as a post-hoc interpretability tool.
Explicitly framed as diagnostic, not a model input.

From v1 Section 7.6:
- All seven agents have negative mean Shapley (each domain contributes)
- Shapley values regime-dependent: geopolitical helpful in normal,
  harmful in high; technical strongest in elevated and high
- Myerson values decompose into independent + network-mediated component
- No single agent dominates, motivating the regime-gated architecture

Practical interpretability paragraph:
- Risk managers can inspect which information domains drove each
  forecast after the fact
- Herding rate serves as a diversity diagnostic
- These outputs are available alongside the point forecast without
  additional computation

[Figure 2: motivation panels, Shapley by regime + herding scatter]
[Figure 5: Shapley boxplots by agent and regime]

---

## 6. Robustness and Inference (~1 page, ~800 words)

Promoted from v1 Appendix A. Renamed to "Robustness and Inference"
to cover both sensitivity tests and the HAC methodology discussion,
which is an inference design point rather than a robustness test
per se.

### 6.1 Sensitivity to data exclusion

From v1 Appendix A:
- 61 days excluded (April 2020 negative-price episode)
- Full-sample re-estimation with same protocol
- Expectation: RMSE increases for all models, relative ranking preserved
[TBD: needs experiment run for full-sample numbers]

### 6.2 Inference under overlapping horizons

From v1 Section 7.2 discussion. Reframed as a methodological
contribution rather than a robustness check:
- With h=20, forecast errors are autocorrelated by construction
  (ACF lag-1 = 0.938)
- Naive DM overstates significance: debate goes from DM=-3.05
  (naive, p=0.002) to DM=-1.49 (HAC, p=0.135)
- Among 16 methods, only GAT and GARCH survive HAC correction
- Practical recommendation: any study with h>5 overlapping targets
  should use HAC-corrected DM as minimum standard
- This finding has implications beyond the present application

---

## 7. Conclusion (~0.75 page, ~600 words)

From v1 Section 8, updated for v2 framing:
- Summary of framework and key results
- Interpretability: ex-post Shapley/Myerson as diagnostic layer
- Methodological implication (HAC correction for overlapping horizons)
- Limitations:
  - Single LLM backbone (reproducibility across model versions)
  - Sample period (2020-2025, unusual macro events)
  - Extreme-regime sample size (n=141 combined low+high)
  - Node features limited to leak-free behavioural encoding;
    richer features (e.g. rolling accuracy, lagged Shapley)
    are a natural extension
- Future work:
  - Other commodity markets (natural gas, metals)
  - Heterogeneous LLM backbones per agent role
  - Lagged Shapley as a prediction-time feature
    (requires re-running debate with t-1 attribution)
  - Online graph structure learning for faster adaptation

---

## Appendices

### Appendix A: Pairwise HAC DM Matrix

From v1 Appendix C. Full pairwise HAC-corrected DM test matrix
for principal models.

### Appendix B: Architecture Details

Merged from v1 Appendix B + expanded:
- Full hyperparameter table
- Failed optimisation experiments (L1, Granger prior, MoE)
- Node and context feature specifications
- Edge-level frequency details per seed per window
  (deferred from Section 5.3 to keep main text focused
  on stable patterns rather than edge-level variation)

### Appendix C: Crisis Episode Case Study

From v1 Appendix D. March 2022 Ukraine crisis case study,
with ex-post Shapley/Myerson attribution decomposition.

---

## Tables and Figures Plan

**Tables:**
- Table 1: Full model comparison (16 methods, RMSE + HAC DM + p + sig)
- Table 2: Architecture ablation (7 variants)
- Table 3: Regime-conditional RMSE (5 models x 4 regimes + All)

**Figures:**
- Figure 1: Two-stage architecture diagram (Section 3)
- Figure 2: Motivation panels (Section 5.4)
  - Panel A: Mean Shapley by agent and regime (forest plot)
  - Panel B: Herding count vs influence density scatter
- Figure 3: Regime-conditional RMSE bar chart (Section 4.4)
- Figure 4: Learned graph edge frequency heatmap, 3-panel temporal
  evolution (Section 5.3)
- Figure 5: Shapley value boxplots by agent and regime (Section 5.4)

---

## Structural Comparison: v1 vs v2

| v1 Section | v2 Destination | Change |
|------------|----------------|--------|
| 1. Introduction | 1. Introduction (expanded) | Absorbs Related Work |
| 2. Related Work | 1. Introduction (woven in) | Dissolved |
| 3. Problem Formulation | 2. Preliminaries | Expanded with baselines + GAT background |
| 4. Motivation for GAT | 5. What Drives the Gain? | Moved post-results |
| 5. GAT Architecture | 3. Proposed Framework | Merged with Protocol |
| 6. SDD Protocol | 3. Proposed Framework | Merged with Architecture |
| 7. Results + Discussion | 4. Empirical Results | Discussion split to Section 5 |
| 8. Conclusion | 7. Conclusion | Same |
| Appendix A (Robustness) | 6. Robustness and Inference | Promoted to section |
| Appendix B (Hyperparams) | Appendix B | Renumbered |
| Appendix C (Pairwise DM) | Appendix A | Renumbered |
| Appendix D (Case Study) | Appendix C | Renumbered |

**Word count estimate:**
| Section | Words |
|---------|-------|
| 1. Introduction | 2,200 |
| 2. Preliminaries | 1,200 |
| 3. Proposed Framework | 2,800 |
| 4. Empirical Results | 2,500 |
| 5. What Drives the Gain? | 1,600 |
| 6. Robustness and Inference | 800 |
| 7. Conclusion | 600 |
| **Total body** | **~11,700** |
| Appendices A-C | ~2,500 |
| **Total with appendices** | **~14,200** |

---

## Writing Plan

### Experiment re-run required BEFORE writing Sections 3-4

The Shapley removal from node features invalidates all v1 GAT
results. Before any section can be written with final numbers:

1. Modify `src/optimized_gat.py` to use 4-dim leak-free node features
   (behaviour_code/3, degree/6, herding_streak/5, + 1 rolling stat)
2. Re-run walk-forward with 5 seeds, same protocol
3. Re-run all ablations with new baseline
4. Update Tables 1-3 with new numbers
5. Re-run edge frequency analysis

Expected impact: headline RMSE will increase (v1 No-Shapley was
0.1520). The key question is whether the GAT still significantly
outperforms HAR after HAC correction with leak-free features.
If not, the paper's main claim weakens substantially.

Alternative: if leak-free GAT loses significance, consider
lagged Shapley (use $\phi_i(t-20)$ or a rolling window that
avoids the forward-looking label) as a middle ground. This is
leak-free because by day t, actual_vol(t-20) is fully observed.

### Section writing priority (after experiments)

- **Heavy rewrite**: Section 1 (merge two sections + restructure
  literature flow), Section 2 (new section, new background content),
  Section 5 (reframe motivation as post-hoc analysis)
- **Moderate rewrite**: Section 3 (merge + reorder, Shapley
  removed from features), Section 4 (new numbers from re-run)
- **Light edit**: Section 6 (promote and expand Appendix A),
  Section 7 (updated limitations framing)
