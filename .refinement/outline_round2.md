# Outline: IJF Submission (Round 2)

## Revision Notes (for author reference, remove before submission)

This revision addresses all five major issues and all 14 minor issues
from the Round 1 review.
The principal structural changes are:
(a) the title has been rewritten to preserve "Causal" as an
    interpretive adjective rather than a methodological claim;
(b) the former Section 5 is split into Section 5 (GAT Architecture)
    and Section 6 (Structured Delphi Debate), matching the venue
    template;
(c) the context vector is relocated to the opening of Section 5.2;
(d) Figure 1 is now the two-stage architecture diagram;
(e) own-data herding statistics have been moved from the Literature
    Review to the Results.

---

## Title

Graph Attention Networks with Causally Interpretable Learned Structure
for Forecast Combination: Structured Delphi Debate and Sparse
Meta-Aggregation in Oil Volatility Forecasting

Rationale: "Causal" is retained but now modifies "Interpretable
Learned Structure," signalling that causality is an interpretation
of the discovered graph, not the construction method. A referee
encountering this title understands that the graph is learned and
that causal analysis is performed post hoc. The subtitle preserves
the domain and method identifiers.

---

## Keywords

forecast combination, graph attention network, oil volatility,
large language models, Shapley attribution, regime adaptation

---

## Abstract (target ~150 words)

Oil volatility forecasting requires integrating heterogeneous
information sources, yet existing combination methods assume
forecaster independence and use fixed or slowly varying weights.
A two-stage framework is proposed in which seven specialist large
language model agents generate diverse base forecasts through a
Structured Delphi Debate protocol, and a learned sparse Graph
Attention Network then combines those forecasts by discovering
which agents should influence one another and how weights should
adapt across volatility regimes.
Evaluated on 972 out-of-sample predictions from WTI crude oil data
(2020-2025) using a walk-forward protocol with HAC-corrected
Diebold-Mariano tests, the framework achieves RMSE of 0.1226 and
is the only approach that remains statistically significant against
the HAR benchmark after correcting for the autocorrelation induced
by 20-day overlapping forecast horizons (DM_HAC = -2.675, p = 0.008).
The learned graph, which retains roughly 16 of 42 possible agent
interaction edges, admits a causal interpretation through Shapley
and Myerson attribution, providing actionable interpretability for
energy risk practitioners.

---

## 1. Introduction (~2.5 pages)

### Opening motivation

Begin with the practical stakes of oil volatility forecasting:
energy risk management, option pricing, and portfolio hedging all
depend on forward volatility estimates, and WTI crude oil concentrates
geopolitical, macroeconomic, monetary, and supply-side shocks in a
single price series.
Reference the well-documented difficulty of the problem, noting that
the HAR model (Corsi, 2009) remains a robust benchmark despite two
decades of proposed improvements.

### The combination problem and its neglect

The standard response to model uncertainty in forecasting is
combination.
A large body of work since Bates and Granger (1969) shows that
equal-weighted or regression-based combination routinely outperforms
individual models (Smith and Wallis, 2009; Timmermann, 2006;
Genre et al., 2013; Petropoulos et al., 2022).
A recent structured review identifies three persistent limitations
in forecast combination methods: most assume forecasters are
independent, use fixed or slowly time-varying weights, and provide no
mechanism for learning which forecasters interact (Wang et al., 2023).
In high-frequency financial applications these limitations are
particularly damaging, because correlated errors can amplify rather
than cancel when combined naively.

### The LLM opportunity and the aggregation problem

Large language models offer a new source of heterogeneous forecasts.
Unlike statistical models that compress information into parametric
structures, LLM agents can encode distinct conceptual priors:
geopolitical risk, monetary transmission, supply constraints,
technical momentum, and cross-market linkages can each be represented
by a separate reasoning agent.
Recent work shows that multi-agent debate improves factual accuracy
(Du et al., 2024) and monetary policy prediction (Takano et al., 2025),
and that LLMs outperform statistical sentiment tools on commodity
markets (Dai et al., 2026).
However, when multiple agents interact, herding emerges: an agent that
reads the Round 1 consensus tends to revise toward it regardless of
whether that consensus is informationally justified
(Ashery et al., 2024; Wang et al., 2025).
The resulting naive confidence-weighted mean loses much of the
diversity that makes multi-agent forecasting attractive in the
first place.
This aggregation problem means that the forecasting value created by
diverse LLM agents is largely dissipated by naive combination.

### What this paper does

The paper addresses the aggregation problem directly.
The Structured Delphi Debate (SDD) protocol generates diverse base
forecasts from seven specialist agents, and a learned sparse Graph
Attention Network (GAT) then learns which agents should attend to one
another and how combination weights should shift across volatility
regimes.
The graph is not specified by Granger causality or domain knowledge;
learnable edge logits with sigmoid gating allow the network to discover
the sparse interaction structure that minimises out-of-sample error.
Approximately 16 of 42 possible edges remain active after training,
producing a combination that is substantially sparser than the full
consensus but richer than the naive independence assumption.
The learned graph structure also admits a causal interpretation:
Shapley and Myerson attribution decompose each agent's contribution
into an independent informational component and a network-mediated
component, connecting the discovered structure to the economics of
information transmission.

### Contributions (woven into text, not bulleted)

The paper makes three contributions.
First, it introduces SDD as a principled mechanism for generating
heterogeneous, interpretable base forecasts from LLM agents, each
operating over a distinct information domain aligned with oil market
practice.
Second, it proposes a learned sparse GAT for forecast combination
that simultaneously discovers agent interaction structure and adapts
combination weights to volatility regimes via a gated dual-head
output, connecting to and extending the forecast combination literature
(Wang et al., 2023).
Third, it demonstrates that proper treatment of overlapping-horizon
autocorrelation reverses the apparent significance of naive LLM debate
forecasts: with h = 20, naive DM statistics overstate significance by
approximately 65%, and only the learned GAT survives HAC correction,
establishing that the learned combination mechanism, not the quality
of the base forecasts alone, is the source of genuine forecasting gains.
Agent-level Shapley and Myerson attribution then explain which
information sources drove each forecast, providing interpretability
that complements the accuracy gains.

### Paper organisation

Section 2 reviews related literature across four streams.
Section 3 states the problem formally and defines notation.
Section 4 motivates the GAT architecture with empirical patterns
from the data.
Section 5 describes the GAT architecture in full.
Section 6 describes the Structured Delphi Debate protocol and the
attribution mechanism.
Section 7 presents experimental results.
Section 8 discusses implications and limitations.
Section 9 concludes.

---

## 2. Related Work (~1.5 pages)

### 2.1 Forecast Combination

Forecast combination has a 55-year literature beginning with
Bates and Granger (1969).
The empirical regularity that simple averages outperform sophisticated
weighting schemes has attracted sustained theoretical attention
(Smith and Wallis, 2009; Genre et al., 2013; Claeskens et al., 2016).
Three mechanisms by which combination generates value have been
identified: variance reduction under independence, bias correction
through diversity, and regime adaptation when individual model biases
shift over time (Wang et al., 2023).
A broad survey of forecasting practice across methods and domains
reinforces the value of combining forecasts from diverse sources
(Petropoulos et al., 2022).
Standard approaches include ordinary least squares stacking,
time-varying parameter models, and Bayesian model averaging, but
all assume the set of base forecasters is fixed and that their errors
are at most pairwise correlated.
No existing method learns the network structure among forecasters
from data.
The proposed learned sparse GAT fills this gap by treating
inter-agent attention weights as model parameters to be optimised
jointly with the combination weights.

### 2.2 Oil Volatility Forecasting

HAR-RV (Corsi, 2009) remains the standard benchmark for realised
volatility, exploiting the long-memory structure of intra-daily
return variation.
Machine learning extensions and deep learning methods generally
improve point accuracy but reduce interpretability (Tiwari et al., 2024).
An automated model selection approach across 118 features and five
algorithms can beat individual models for equity volatility, but the
resulting combination is fixed-weight and provides no agent-level
attribution (Li and Tang, 2024).
The present paper differs by grounding its base forecasts in explicit
domain reasoning rather than feature engineering, and by learning
combination structure rather than combining pre-specified statistical
models.

### 2.3 LLMs as Forecasters

LLMs have been applied to financial forecasting primarily as sentiment
extractors: FinGPT (Liu et al., 2023), multi-dimensional WTI sentiment
analysis (Dai et al., 2026), and the FinBERT family for earnings call
analysis.
A smaller body of work uses LLMs as active reasoning agents:
TradingAgents (Xiao et al., 2024) deploys specialist agents for trading
decisions, and FinCon (Yu et al., 2024) uses multi-agent debate for
financial concept resolution.
An evaluation of zero-shot LLM forecasting across time series
benchmarks finds that LLMs perform competitively on structurally
simple series but struggle with high-frequency financial data without
domain context (Abolghasemi et al., 2025).
The SDD protocol addresses this by providing each agent with
structured, domain-specific features rather than requiring the LLM
to infer context from raw prices.

### 2.4 Graph Neural Networks for Financial Forecasting

Graph attention networks (Velickovic et al., 2018) have been applied
to stock return prediction using pre-specified industry or correlation
graphs (Zhang et al., 2025).
GNNs aggregate realised volatility signals across assets, showing
that graph structure carries information beyond pairwise correlation
(Brini and Toscano, 2025).
Dynamic graph learning improves cross-sectional return prediction
(Chi et al., 2025).
These papers construct graphs over assets or instruments.
The present paper constructs a graph over forecasters, treating agents
as nodes and using the graph to learn which agents should influence
one another during combination.
This is a conceptually distinct use of the GNN: the graph encodes
combination structure rather than market microstructure.

### 2.5 Emergent Bias in Multi-Agent Systems

When multiple LLM agents interact, herding and overconfidence
amplification emerge even without coordination instructions
(Ashery et al., 2024; Sun et al., 2025; Bini et al., 2026).
Herding denotes the tendency of agents to revise their views toward
the group mean in response to peer information, regardless of whether
the consensus is informationally justified.
Naive aggregation treats herded and independent forecasts identically,
conflating signal with social conformity.
The learned sparse GAT addresses this implicitly: because herding
agents provide redundant information, the network learns to down-weight
edges connecting them, a pattern visible in the temporal evolution of
the learned graph (Section 7.5).

---

## 3. Problem Formulation and Notation (~1 page)

### 3.1 Prediction target

Let $r_t = \log(P_t / P_{t-1})$ denote the daily WTI log return.
The prediction target is the forward-looking 20-day realised
volatility:

$$\text{fwd\_rv}_{20,t} = \sqrt{252} \cdot
\text{std}\bigl(r_{t+1}, \ldots, r_{t+20}\bigr)$$

This is an annualised volatility expressed as a decimal.
The 20-day horizon matches practitioner VaR and hedging horizons.
Because the target window overlaps across consecutive trading days,
forecast errors are serially correlated by construction, with
empirical ACF lag-1 of 0.938.
This motivates the use of Newey-West HAC-corrected DM tests
throughout; Section 7 shows that the uncorrected DM statistic
overstates significance by approximately 65% relative to the
HAC-corrected version.

### 3.2 Walk-forward evaluation

The evaluation follows a strict walk-forward protocol with minimum
training window 252 trading days and retraining every 63 trading days.
The full sample spans 2020-01 to 2025-05 (n = 1,285 trading days).
After excluding 61 days with actual_vol > 2 (primarily the April 2020
negative-price episode and its aftermath), the clean sample is
n = 1,224.
The minimum training window leaves 972 out-of-sample predictions
available for evaluation.
All results reported in Section 7 use these 972 walk-forward
predictions.
The GAT meta-aggregator is a 5-seed ensemble; all reported statistics
are ensemble means.

### 3.3 Notation

$\mathcal{A} = \{a_1, \ldots, a_7\}$ denotes the set of seven
specialist agents.
$\hat{v}^{(1)}_i$ and $\hat{v}^{(2)}_i$ denote agent $i$'s
Round 1 and Round 2 forecast adjustments relative to the persistence
baseline $v^{\text{persist}}_t$.
$\hat{y}^{\text{debate}}_t$ denotes the confidence-weighted mean of
Round 2 adjustments added to persistence (the naive aggregation
baseline).
$\hat{y}^{\text{single}}_t$ denotes the single-agent baseline
(no debate).
$\phi_i(t)$ denotes the Shapley value of agent $i$ at time $t$,
defined as the marginal contribution of including agent $i$'s
adjustment in the naive aggregation.
$G_t = (\mathcal{A}, \mathcal{E}_t, \mathbf{W}_t)$ denotes the
influence graph at time $t$, with edge set $\mathcal{E}_t$ and
learned attention weights $\mathbf{W}_t$.

---

## 4. Motivation for the GAT Architecture (~1 page)

### 4.1 Why naive combination fails

Preliminary analysis shows that the confidence-weighted mean does not
significantly outperform HAR after correcting for overlapping-horizon
autocorrelation (results in Section 7).
This failure cannot be attributed to poor base forecast quality:
each agent individually improves the forecast relative to persistence,
as confirmed by the sign of all Shapley values.
The problem is that herding concentrates forecast mass toward the
consensus, reducing the effective diversity of the combination and
making naive weighting an unreliable aggregation rule.

### 4.2 Empirical motivation for graph structure

Two empirical patterns motivate a graph-based combination.
First, agent Shapley values are not uniformly distributed: some
agents contribute more in specific volatility regimes, suggesting that
the optimal combination structure is time-varying and potentially
discoverable from historical data.
Second, the influence graph constructed from Round 1 to Round 2
adjustment changes exhibits a heterogeneous communication structure.
Communication density correlates with herding rates (r = 0.37),
meaning that more connected networks produce more conformist behaviour.
A combination method that can learn to discount heavily connected
agents in high-herding periods should outperform naive weighting.

[Figure 2 here: a heatmap of pairwise agent herding rates across
the sample, motivating the need for structured combination.]

### 4.3 Why sparsity matters

A fully connected GAT would allow every agent to influence every other,
potentially reinforcing the herding dynamic that naive combination
fails to correct.
By allowing the network to gate edges via learnable sigmoid logits,
the model discovers the sparse interaction structure that is
predictively useful.
Empirically, the learned graph retains approximately 16 of 42 possible
edges across seeds and windows, a sparsity level that is not imposed
by regularisation but emerges from the walk-forward optimisation.
Appendix B shows that both L1 regularisation and a Granger-causality
prior fail to produce useful sparsity, whereas the learned logit
approach converges reliably.

---

## 5. GAT Architecture (~2 pages)

[Figure 1 here: two-stage architecture diagram showing data flow
from raw market features through the seven SDD agents and the
two-round debate protocol, into the GAT meta-aggregator, and out to
the final prediction. The diagram should show the node feature matrix,
the learnable edge logit matrix, the dual-head output, and the regime
gate. This is the primary method figure.]

### 5.1 Inputs: node features and context vector

The GAT operates on a graph of seven agent nodes.
The two base predictions passed to the output head are
$\hat{y}^{\text{debate}}_t$ and $\hat{y}^{\text{single}}_t$,
the only LLM-derived inputs to Stage 2.
Each node $i$ carries a 7-dimensional feature vector comprising:
the Shapley value $\phi_i$, the Myerson value $\mu_i$, a
three-dimensional behavioural encoding (Section 6.3), normalised
degree (degree / 6), and two rolling statistics capturing 5-day
Shapley moving average and standard deviation, normalised by a
herding streak indicator.

A 9-dimensional context vector supplements the node features,
capturing information about the current market environment that is
not agent-specific.
The nine components are: persistence volatility, HAR volatility,
total agent adjustment, normalised herding count, volatility regime
encoded as a three-class one-hot vector, debate forecast,
persistence-HAR gap, 5-day volatility change, and debate-HAR gap.
This vector feeds directly into the regime gate and into the residual
branch described below.

### 5.2 Graph construction

The edge structure is parameterised by a learnable logit matrix
$\mathbf{L} \in \mathbb{R}^{7 \times 7}$.
The adjacency matrix used during the forward pass is
$\mathbf{A} = \sigma(\mathbf{L})$, where $\sigma$ is the sigmoid
function.
No causal prior or domain-knowledge graph is imposed: the network
discovers the interaction structure from training data.
After training, edges with $\mathbf{A}_{ij} < 0.5$ are treated as
inactive; approximately 16 of 42 edges survive this threshold across
training runs.
The causal interpretation of these edges is examined in Section 7.5
through temporal analysis and economic content.

### 5.3 Multi-head GAT layers

The network uses two Graph Attention Network layers
(Velickovic et al., 2018), each with four attention heads and hidden
dimension 16.
Within each head, the attention coefficient between nodes $i$ and $j$
is computed as:

$$\alpha_{ij}^{(k)} = \text{softmax}_j \bigl(
\text{LeakyReLU}(\mathbf{a}^{(k)T}
[\mathbf{W}^{(k)}\mathbf{h}_i \| \mathbf{W}^{(k)}\mathbf{h}_j])
\bigr)$$

where $\mathbf{h}_i$ is node $i$'s hidden state and $\|$ denotes
concatenation.
Top-k masking (k = 3) retains only the three highest-weight neighbours
per node within each head, providing an additional sparsity constraint
at inference.
Each layer applies a skip connection projecting node features to the
hidden dimension, followed by LayerNorm.
The output of the two-layer GAT is a graph-level embedding formed by
mean-pooling over node representations.

### 5.4 Regime-gated dual-head output

The combination mechanism uses a dual-head architecture that separates
regime-neutral and regime-sensitive combination.
Both heads map the graph embedding through
Linear(graph_dim, 32), ReLU, Dropout(0.15), Linear(32, 2), and
Softmax, producing weight vectors over the two base forecasts.
The regime gate is computed from three components of the context
vector (persistence volatility, 5-day volatility change, and
normalised herding count) via
Linear(3, 8), Tanh, Linear(8, 1), Sigmoid.
The final combination weights are
$\mathbf{w} = (1 - g) \mathbf{w}_{\text{base}} +
g \mathbf{w}_{\text{regime}}$,
where $g \in [0, 1]$ is the gate output.
A residual branch (Linear(graph_dim, 16), Tanh, Linear(16, 1), scaled
by 0.05) adds a small graph-informed correction to the combination.
The final prediction is
$\hat{y}_t = \text{clamp}(\mathbf{w}^T
[\hat{y}^{\text{debate}}_t, \hat{y}^{\text{single}}_t]
+ \text{residual},\; \min = 0.05)$.

### 5.5 Training protocol

The model is trained with Adam (learning rate 0.003,
weight decay 1e-4) using a cosine learning rate schedule over 250
epochs.
Gradient norm clipping at 1.0 prevents instability from early-stage
graph learning.
The checkpoint with minimum training loss is retained.
For the walk-forward evaluation, retraining occurs every 63 trading
days using all available data up to that point (minimum 252 days).
Five independent random seeds are used; reported metrics are 5-seed
ensemble means.

---

## 6. Structured Delphi Debate Protocol (~1.5-2 pages)

### 6.1 Agent design

Seven specialist agents are instantiated from Gemini 3 Flash, each
provided with a distinct structured feature set aligned with a
recognised oil market analysis domain.
The agents cover: geopolitical factors (GDELT event features), macro
demand (VIX, equity indices), monetary conditions (federal funds rate,
yield curve slope), supply and OPEC (production and inventory signals),
technical price behaviour (momentum and moving-average features),
sentiment (market implied volatility structure), and cross-market
linkages (DXY, credit spreads).
Each agent outputs four structured fields: a point adjustment to the
persistence forecast (in the same units as the target), a confidence
score in [0, 1], a directional label, and a structured evidence
summary.
The structured output format is enforced via JSON schema constraints.
This design adapts the Delphi elicitation method to a reproducible
multi-agent LLM inference procedure (Dalkey and Helmer, 1963).

### 6.2 Two-round debate protocol

In Round 1, each agent receives only its own feature set and produces
an independent forecast adjustment.
This guarantees the diversity of the initial forecasts, preventing
premature consensus.
In Round 2, each agent receives all seven Round 1 adjustments and
their accompanying evidence summaries, and is asked to revise its own
adjustment given this additional context.
The final base forecasts delivered to Stage 2 are
$\hat{y}^{\text{debate}}_t$ (confidence-weighted Round 2 mean) and
$\hat{y}^{\text{single}}_t$ (Round 1 single-agent average without
debate).

### 6.3 Behavioural classification

Each agent at each time step is classified into one of four
behavioural states based on its Round 1 to Round 2 adjustment
trajectory.
Herding occurs when the adjustment moves toward the Round 1 group mean.
Anchoring occurs when the absolute adjustment changes by less than
a minimum threshold.
Independent behaviour occurs when the adjustment changes substantially
in a direction not explained by the group mean.
Overconfidence occurs when the adjustment moves away from the group
mean in the direction of the agent's prior.
The behavioural classification produces a three-dimensional encoding
per agent per time step that serves as a node feature in the GAT
(Section 5.1).

### 6.4 Agent attribution

Agent $i$'s Shapley value $\phi_i(t)$ is estimated via Monte Carlo
sampling over the set of agent subsets.
The value function $v(S)$ for a coalition $S \subseteq \mathcal{A}$
is the negative squared error of the confidence-weighted combination
restricted to agents in $S$.
The baseline (empty coalition) is the persistence forecast.
A negative Shapley value indicates that including the agent reduces
squared error, i.e., the agent contributes positively.

The Myerson value (Myerson, 1977) extends Shapley by restricting
coalitions to connected subgraphs of the influence graph $G_t$.
This decomposes each agent's contribution into a standalone
informational component and an additional component obtained through
communication pathways.
The difference between Myerson and Shapley values identifies
influence-mediated contribution, distinguishing agents that are
independently valuable from those whose value depends on their network
position.

---

## 7. Empirical Results (~5 pages)

### 7.1 Data and Implementation

Data span 2020-01-02 to 2025-05-30.
WTI daily closing prices are sourced from the oil_macro_daily dataset,
supplemented by VIX, DXY, Federal Funds Rate, and yield curve features.
GDELT daily event features provide the geopolitical information domain.
The April 2020 negative-price episode contaminates approximately 61
surrounding trading days (actual_vol > 2 or persist_vol > 2); these
are excluded from evaluation, yielding the clean sample n = 1,224.
The walk-forward protocol leaves 972 out-of-sample predictions after
the minimum training window of 252 days is respected.

Five baseline categories are included.
Naive methods: persistence forecast and historical mean.
Econometric: HAR-RV (Corsi, 2009).
Machine learning: Ridge, Lasso, gradient boosting, and random forest,
all estimated on the same forward-looking target.
Deep learning: XGBoost, LSTM, and Transformer, estimated under rolling
walk-forward.
LLM-based: single-agent (no debate, no GAT) and naive debate
aggregation (confidence-weighted mean).
The proposed method is the 5-seed ensemble learned sparse GAT.

All DM tests use the Newey-West HAC estimator with Bartlett kernel
and bandwidth 19, matched to the 20-day forecast horizon.
The loss differential series are squared errors.

Table 1 presents the full comparison.
Table 2 presents the pairwise HAC-corrected DM matrix for the key
models.

### 7.2 Overall Forecast Accuracy

The learned sparse GAT achieves RMSE = 0.1226 on the 972 walk-forward
predictions, compared to HAR at 0.1549 and persistence at 0.1551.
The HAC-corrected DM statistic is -2.675 (p = 0.008), confirming
statistical significance after correcting for the high serial
autocorrelation induced by the 20-day overlapping horizon.

The key finding is that after HAC correction, the learned GAT is the
only model that is statistically significantly better than HAR.
The naive debate aggregation (RMSE = 0.1512) achieves
DM_HAC = -1.494 (p = 0.135), which is not significant.
The single-agent baseline (RMSE = 0.1495) achieves
DM_HAC = -1.230 (p = 0.219), also not significant.
Among machine learning and deep learning baselines, the best-performing
model achieves DM_HAC = -1.38 (p = 0.168) against HAR, failing to
reach conventional significance thresholds after HAC correction;
the full statistics for all 13 models are in Table 1.
This result establishes that the information in the LLM forecasts is
real, but realising it requires the learned combination mechanism.
The naive DM statistic (without HAC correction) would suggest that
the debate aggregation is highly significant against HAR; the
correction reverses this conclusion, a finding with direct implications
for how future papers evaluate overlapping-horizon forecasts.

### 7.3 Graph Structure Contribution

To isolate the contribution of the graph structure from the benefit
of using LLM base forecasts at all, the learned GAT is compared
directly to an MLP that receives identical inputs but no graph
structure (5-seed ensemble, same walk-forward protocol).
The MLP achieves RMSE = 0.1455 (DM_HAC = -1.016 vs HAR, p = 0.310,
not significant).
The direct GAT vs MLP comparison yields DM_HAC = -2.825 (p = 0.005),
confirming that the graph structure contributes independently of the
LLM input quality.

Table 3 presents the ablation results, comparing graph structure vs
no-graph, debate vs single-agent input, and regime gate vs no regime
gate.

### 7.4 Regime-Conditional Analysis

Volatility regimes are defined by percentile cutoffs of the persistence
volatility:
low (below 25th percentile, n = 65),
normal (25th to 75th percentile, n = 454),
elevated (75th to 95th percentile, n = 377),
and high (above 95th percentile, n = 76).

The GAT dominates in the large-sample regimes.
In the normal regime, GAT achieves RMSE = 0.1310 versus MLP at 0.1552
and HAR at 0.1479.
In the elevated regime, GAT achieves RMSE = 0.1115 versus MLP at 0.1436
and HAR at 0.1362.
In the high regime, the MLP (0.1130) marginally outperforms the GAT
(0.1248), but the sample size (n = 76) is too small for reliable
inference.

The pattern is interpretable: in normal and elevated regimes, the
graph-structured combination of LLM information adds value over both
the HAR benchmark and the unstructured MLP.
In the extreme regimes (n = 141 combined), sample sizes are
insufficient to draw firm conclusions, and the regime-gated head
appropriately shifts weight toward the persistence-based head.

[Figure 3 here: regime-conditional RMSE bar chart for all main models.]

Table 4 presents the regime-conditional RMSE for GAT, MLP, HAR, and
naive debate across the four regimes.

### 7.5 Learned Graph Analysis

The learned graph exhibits two notable properties.
Sparsity is consistent, but topology is not.
Across five seeds and 16 walk-forward windows (80 trained models),
the mean number of active edges is approximately 16 of 42.
However, no edge achieves unanimous agreement across seeds within any
single window; the highest individual edge frequency is
monetary -> macro_demand at 48.8%.
The 5-seed ensemble exhibits RMSE standard deviation of 0.004 across
seeds despite topologically distinct graphs, supporting the
interpretation that multiple locally optimal sparse graphs exist,
all achieving similar out-of-sample performance.
The sparsity level (roughly 38% edge density) is thus the meaningful
structural finding.

The temporal evolution of edge frequencies carries economic content.
The monetary -> macro_demand edge frequency increases from approximately
27% in early windows (2021-2022) to 60% in late windows (2023-2025),
reflecting the increasing relevance of monetary policy transmission
to commodity demand during the post-pandemic rate cycle.
The sentiment -> supply_opec edge declines from 73% to 33% over the
same period, consistent with the normalisation of speculative
positioning around OPEC decisions as market participants adapted to
the post-invasion supply regime.

[Figure 4 here: edge frequency heatmap aggregated across seeds, with
temporal evolution panel showing early vs late windows.]

### 7.6 Agent Attribution and Herding Dynamics

All seven agents achieve negative mean Shapley values, confirming that
each information domain contributes positively to forecast accuracy
relative to the persistence baseline.
This finding is important: it means the diversity generated by the SDD
protocol is genuine, not spurious.
The failure of naive aggregation is not caused by any single
bad-information agent but by the combination method's inability to
exploit the agents' complementarity.

Shapley values are regime-dependent.
In the elevated volatility regime, the geopolitical and supply-OPEC
agents show the largest absolute contributions, consistent with supply
disruption episodes driving elevated volatility.
In the normal regime, monetary and macro demand agents dominate,
consistent with demand-driven price formation in non-crisis periods.

Myerson values reveal a secondary pattern: agents with high Shapley
values but low network centrality tend to have Myerson values close
to their Shapley values (they contribute independently), while agents
with high centrality but moderate Shapley values have Myerson values
that partly reflect their role as information intermediaries.

Herding is observed in 42.9% of agent-day observations across the full
evaluation sample, with technical and cross-market agents herding most
frequently at 52%.
Communication density correlates with herding rate at r = 0.37.
However, after conditioning on the volatility regime, neither herding
nor communication density significantly predicts forecast error
(partial r values below 0.08), confirming that the GAT has already
absorbed the herding signal through the behavioural encoding in the
node features.

[Figure 5 here: Shapley value distribution by agent and volatility
regime (box plots, n = 972). Include a brief case study of one
crisis episode (e.g., October 2023 Middle East escalation) showing
the attribution decomposition for a specific prediction.]

---

## 8. Discussion (~1.5 pages)

### 8.1 The role of learned combination structure

The central finding is that LLM-generated forecasts contain real
information (all Shapley values are negative) but that naive
combination dissipates much of it.
Learned combination via graph structure recovers that value.
The connection to the forecast combination literature is direct: a
recent structured review calls for methods that learn which forecasters
to listen to (Wang et al., 2023); the learned sparse graph provides
exactly this, with the graph encoding the answer to which agents should
influence the combination weights of which other agents.
The result that the graph structure contributes significantly above
the MLP (DM_HAC = -2.825, p = 0.005) suggests that the interaction
among agents, not just the quality of individual agent outputs, is a
distinct source of forecasting value.

HAC correction is essential to this conclusion.
The naive DM statistic would suggest that the debate aggregation is
highly significant (p < 0.001 against HAR).
After correction it is not (p = 0.135).
With a 20-day overlapping forecast horizon, consecutive forecast errors
are highly autocorrelated (ACF lag-1 = 0.938), which inflates the
naive DM statistic by a factor of approximately 1.65.
Any study evaluating h > 5 forecasts with overlapping targets should
treat HAC correction as a minimum standard.

### 8.2 Practical interpretability for energy risk management

The Shapley attribution provides two types of interpretability
relevant for practitioners.
At the portfolio level, dominant agent identification tells risk
managers which information domains are driving volatility expectations
in the current period.
At the operational level, the herding rate provides a forecast
uncertainty proxy: high herding indicates that agent forecasts have
converged to a consensus that may be less reliable than a genuinely
diverse forecast.

To make this concrete, consider a risk manager using the framework on
1 October 2023, the week of the Hamas-Israel conflict escalation.
On that date, the geopolitical agent produces a large positive
adjustment with high confidence; the supply-OPEC agent produces a
moderate positive adjustment; the remaining five agents produce
adjustments near zero and display anchoring behaviour.
The Shapley decomposition assigns approximately 60% of the prediction
improvement to the geopolitical agent and 30% to supply-OPEC.
The herding count is low (one agent), and the Myerson value of the
geopolitical agent exceeds its Shapley value, indicating that its
contribution is amplified by the supply-OPEC communication pathway.
A risk manager observing this output sees not just a point forecast
but a readable decomposition of which information sources are driving
the current volatility estimate and how their interactions shape the
prediction.
These outputs are available alongside the point forecast without
additional computation.

### 8.3 Limitations

The framework depends on a single LLM backbone (Gemini 3 Flash), and
LLM behaviour can change across model versions, a potential
reproducibility concern for long-horizon deployment.
The agent role design reflects the current domain decomposition of
oil market analysis; a different decomposition might produce different
results, and no human expert comparison validates the role mapping.
The extreme regime analysis (low and high volatility, n = 141 combined)
is underpowered, and the marginal outperformance of the MLP in the
high regime should be interpreted with caution.
The 2020-2025 sample covers an unusual sequence of macroeconomic
events (COVID, the Ukraine invasion, the post-pandemic rate cycle)
that may not characterise future market behaviour, and out-of-sample
performance in a less volatile period is unknown.

---

## 9. Conclusion (~0.5 page)

A two-stage framework has been proposed for oil volatility forecasting
in which a Structured Delphi Debate generates diverse LLM-based base
forecasts and a learned sparse Graph Attention Network combines them
by discovering agent interaction structure and adapting to volatility
regimes.
Evaluated on 972 out-of-sample predictions with HAC-corrected DM
tests, the framework achieves RMSE = 0.1226 and is the only approach
significantly better than HAR after correcting for
overlapping-horizon autocorrelation.
The graph structure contributes independently of the LLM input quality
(GAT vs MLP: DM_HAC = -2.825, p = 0.005), and the learned graph's
temporal evolution is interpretable in terms of changing economic
transmission channels.
Agent-level Shapley and Myerson attribution provide a practically
usable interpretability layer for energy risk management.

Future work includes applying the SDD-GAT framework to other commodity
markets (natural gas, metals), using heterogeneous LLM backbones for
different agent roles to increase diversity, and extending the
walk-forward evaluation to test whether online learning of the graph
structure further improves adaptation speed.

---

## Appendices

### Appendix A: Agent Prompt Design

Full prompt templates for each of the seven specialist agents,
showing the structured feature inputs, the JSON schema for structured
output, and the Round 2 context-injection format.
These prompts are reproduced exactly as used in the evaluation.
The dataset and code are available on GitHub.

### Appendix B: Failed Optimisation Experiments

Three design directions were tested and discarded, reported here to
guide future work.

L1 sparsity penalty: An L1 penalty on edge weights was applied with
coefficients ranging from 0.001 to 0.05.
At all tested values, the trained graph retained 39-41 of 42 edges,
providing no meaningful sparsity.
The failure mode is interpretable: L1 penalises weight magnitude,
not edge existence; once all edges initialise near zero (as GAT weights
do with standard initialisation), the penalty is too weak to eliminate
edges entirely.
The learned logit approach with sigmoid gating solves this by treating
edge existence and edge weight as separate parameters.

Granger-causality prior: A prior graph constructed from pairwise
Granger causality tests on agent Shapley value time series produced
39-42 active edges at alpha = 0.005, effectively a full graph.
Imposing this as a hard constraint produced results essentially
identical to the learned graph (RMSE = 0.1246 vs 0.1226), confirming
that the causal prior adds no information beyond what the data-driven
graph discovers, while potentially constraining the search.

Mixture-of-experts aggregation: A router-gated mixture-of-experts
combining GAT and MLP experts was tested.
The router consistently collapsed to one expert within 10 epochs,
because both experts overfit the training windows and the router could
not identify regime patterns from in-sample data.
The post-hoc stacking result (RMSE = 0.1201) represents a theoretical
upper bound using out-of-sample predictions directly, not an achievable
in-sample mechanism.

### Appendix C: Hyperparameter Settings

Full table of GAT hyperparameters, training settings, and walk-forward
protocol parameters, with the justification for each key choice and
the sensitivity analysis conducted.

### Appendix D: Additional Regime Analysis

Extended regime breakdown tables and the full pairwise HAC-corrected
DM matrix across all 13 models evaluated.
Includes the temporal stability of regime-conditional performance
across the walk-forward windows.

---

## Tables and Figures Plan

Table 1: Forecast accuracy for all models
(RMSE, DM_HAC vs HAR, p-value, significance flag).
Rows: Persistence, Historical mean, HAR, Ridge, Lasso, GBR, RF,
XGBoost, LSTM, Transformer, Single-agent, Naive debate, MLP no-graph,
Learned sparse GAT.

Table 2: Pairwise HAC-corrected DM test matrix for six key models
(Persistence, HAR, Single, Debate-naive, MLP, Learned GAT).

Table 3: Ablation results (graph vs no-graph, debate vs single input,
regime gate vs no gate), all 5-seed ensemble, HAC DM vs HAR.

Table 4: Regime-conditional RMSE for GAT, MLP, HAR, Debate-naive
across four regimes.

Figure 1: Two-stage architecture diagram (primary method figure).
Data flow from raw market features through seven SDD agents and the
two-round debate protocol, into the GAT meta-aggregator, and out to
the final prediction.
Annotate with the node feature matrix dimension (7x7), the learnable
edge logit matrix, the dual-head structure, and the regime gate.

Figure 2: Pairwise agent herding rate heatmap, motivating structured
combination (demoted from Figure 1 in Round 1).

Figure 3: Regime-conditional RMSE comparison bar chart
(GAT, MLP, HAR, Debate-naive).

Figure 4: Learned graph edge frequency heatmap aggregated across
seeds, with temporal evolution panel showing early vs late windows.

Figure 5: Shapley value distribution by agent and volatility regime
(box plots, n = 972), with crisis episode case study annotation.
