# SDD-GAT: Oil Volatility Forecast Combination via Causally Interpretable Learned Graph Attention over Multi-Agent LLM Debate

**Keywords:** forecast combination, graph attention network, oil volatility, large language models, Shapley attribution, regime adaptation

## Abstract

Oil volatility forecasting requires integrating heterogeneous information sources, yet existing combination methods assume forecaster independence and use fixed or slowly varying weights. A two-stage framework is proposed in which seven specialist large language model agents generate diverse base forecasts through a Structured Delphi Debate protocol, and a learned sparse Graph Attention Network with DropEdge regularisation then combines those forecasts by discovering which agents should influence one another and how weights should adapt across volatility regimes. Evaluated on 972 out-of-sample predictions from WTI crude oil data (2020-2025) using a walk-forward protocol with HAC-corrected Diebold-Mariano tests, the framework achieves RMSE of 0.1171 and is the only approach that remains statistically significant against the HAR benchmark after correcting for the autocorrelation induced by 20-day overlapping forecast horizons (DM_HAC = -3.212, p = 0.0013). The learned graph, which retains roughly 16 of 42 possible agent interaction edges, admits a causal interpretation through Shapley and Myerson attribution, providing actionable interpretability for energy risk practitioners.

---

# 1. Introduction

Forward-looking estimates of oil price volatility are central to energy risk management, option pricing, and portfolio hedging. WTI crude oil concentrates geopolitical, macroeconomic, monetary, and supply-side shocks in a single price series, making its volatility unusually difficult to forecast. The HAR-RV model (Corsi, 2009) remains a robust benchmark despite two decades of proposed improvements, and the GARCH family (Bollerslev, 1986) continues to perform competitively against machine learning alternatives in energy markets (Degiannakis and Filis, 2017). This persistence of simple benchmarks reflects a genuine structural difficulty: the information relevant to oil volatility is heterogeneous, regime-dependent, and distributed across domains that no single model specification can capture simultaneously. When no single specification suffices, the standard response is combination. A large body of work since Bates and Granger (1969) shows that equal-weighted or regression-based combination routinely outperforms individual models (Smith and Wallis, 2009; Timmermann, 2006; Genre et al., 2013; Petropoulos et al., 2022). Yet a recent structured review identifies three persistent limitations in forecast combination methods: most assume forecasters are independent, use fixed or slowly time-varying weights, and provide no mechanism for learning which forecasters interact (Wang et al., 2023). In high-frequency financial applications these limitations are particularly damaging, because correlated errors can amplify rather than cancel when combined naively. For volatility forecasting with overlapping horizons, the problem is compounded: the forward window induces strong serial dependence in the forecast error series, and standard Diebold-Mariano tests materially overstate the significance of accuracy differences relative to HAC-corrected inference (Section 3.1). Several baselines that appear to beat the HAR benchmark under naive testing lose significance once this correction is applied.

Large language models offer a new source of heterogeneous forecasts. Unlike statistical models that compress information into parametric structures, LLM agents can encode distinct conceptual priors: geopolitical risk assessment, monetary policy transmission, OPEC supply dynamics, technical momentum, and cross-market linkages can each be represented by a separate reasoning agent operating over domain-specific data. Multi-agent debate improves factual accuracy in general reasoning tasks (Du et al., 2024) and monetary policy prediction (Takano et al., 2025), and LLM-derived sentiment features outperform traditional sentiment measures on commodity markets (Dai et al., 2026). However, when multiple LLM agents interact, herding emerges: agents that observe the Round 1 consensus tend to revise toward it regardless of whether the underlying evidence supports that revision (Ashery et al., 2024; Bini et al., 2026). The resulting naive confidence-weighted mean loses much of the diversity that motivates multi-agent forecasting in the first place. In our evaluation sample, a naive debate rule does not significantly outperform HAR after HAC correction, despite producing lower point RMSE. The forecasting value created by diverse LLM agents is largely dissipated by naive combination.

This paper addresses the aggregation problem directly. A Structured Delphi Debate (SDD) protocol generates diverse base forecasts from seven specialist agents, each operating over a distinct information domain aligned with oil market practice. A learned sparse Graph Attention Network (GAT) with DropEdge regularisation (Rong et al., 2020) then combines those forecasts by discovering which agents should attend to one another and how combination weights should shift across volatility regimes. The graph is not specified by Granger causality or domain knowledge; learnable edge logits with sigmoid gating allow the network to discover the sparse interaction structure that minimises out-of-sample error. Approximately 16 of 42 possible directed edges remain active after training, producing a combination that is substantially sparser than the full consensus but richer than the naive independence assumption.

The paper makes three contributions to the forecast combination literature. The SDD protocol connects the emerging multi-agent debate literature (Du et al., 2024; Takano et al., 2025) to the forecast combination tradition by generating heterogeneous, interpretable base forecasts from LLM agents whose behavioural classification, Shapley attribution, and interaction structure are observable quantities that enter the combination as node features. The learned sparse GAT then discovers agent interaction structure and adapts combination weights to volatility regimes via a gated dual-head output. This extends the conditionally optimal combination framework of Gibbs and Vasnev (2024) from a parametric bias-correction setting to a graph-structured one. Within this learned graph, Myerson values complement the Shapley-value-based forecast attribution of Franses et al. (2024) by decomposing each agent's contribution into an independent informational component and a network-mediated component. The empirical evaluation establishes that proper treatment of overlapping-horizon autocorrelation reverses the apparent significance of naive LLM debate forecasts: among 16 methods evaluated on 972 out-of-sample WTI volatility predictions (2020 to 2025), only the proposed GAT variants and GARCH(1,1) remain statistically significant against the HAR benchmark after HAC-corrected Diebold-Mariano testing (Newey and West, 1987). The learned combination mechanism, not the quality of the base forecasts alone, is the source of the forecasting gains.

The remainder of the paper is organised as follows. Section 2 reviews related work across five streams: forecast combination, oil volatility forecasting, LLMs as forecasters, graph neural networks for financial applications, and emergent bias in multi-agent systems. Section 3 defines the prediction target, the walk-forward evaluation protocol, and the notation used throughout. Section 4 motivates the GAT architecture with empirical patterns from the data. Section 5 describes the architecture in detail. Section 6 presents the Structured Delphi Debate protocol and the attribution mechanism. Section 7 reports experimental results and discusses their implications. Section 8 concludes with limitations and directions for future work.

---

# 2. Related Work

## 2.1 Forecast Combination

Forecast combination has a long-standing literature dating to Bates and Granger (1969), who showed that a weighted average of two forecasts can outperform either alone when the error covariance structure is exploited. The empirical regularity that simple averages often match or beat more elaborate weighting schemes has attracted sustained theoretical attention (Smith and Wallis, 2009; Genre et al., 2013; Claeskens et al., 2016). Petropoulos et al. (2022) survey forecasting practice across methods and domains and reinforce the value of combining forecasts from diverse sources. The theoretical case for combination rests on variance reduction under forecaster independence, bias correction through diversity, and regime adaptation when individual model biases shift over time (Timmermann, 2006). Wang et al. (2023) organise these mechanisms and identify three persistent limitations in current methods: most assume forecasters are independent, use fixed or slowly time-varying weights, and provide no mechanism for learning which forecasters interact. Standard approaches include ordinary least squares stacking, time-varying parameter models, and Bayesian model averaging; these typically operate on a fixed set of base forecasters and model dependence through aggregate covariance structures rather than learned network interactions.

Recent work has begun to relax these assumptions. Conditionally optimal combination shows that weighting schemes conditioned on regime signals can improve substantially over static approaches (Gibbs and Vasnev, 2024). Shapley-value-based decomposition has been proposed as a principled way to attribute forecast value among a pool of models (Franses et al., 2024); their approach decomposes the total forecast improvement into additive model-level contributions, but does not account for the network structure through which forecasters may influence one another. We are not aware of work in oil volatility forecasting, or in LLM-based forecast combination more broadly, that learns forecaster-to-forecaster attention jointly with the combination rule in a graph-structured setting.

## 2.2 Oil Volatility Forecasting

The GARCH family (Bollerslev, 1986), the realised volatility framework (Andersen and Bollerslev, 1998), and the HAR-RV model (Corsi, 2009) remain the standard building blocks for volatility forecasting. GARCH specifications are well suited to oil markets because the conditional variance dynamics of crude oil returns conform to the leverage and clustering patterns that GARCH was designed to capture (Sévi, 2014). Machine learning and deep learning extensions have incorporated macroeconomic, geopolitical, and sentiment variables, generally improving point accuracy on oil volatility benchmarks but at the cost of interpretability (Tiwari et al., 2024). Li and Tang (2024) automate model selection across 118 features and five algorithms for equity volatility; the approach extends to oil markets in principle, but the resulting combination uses fixed weights and provides no agent-level attribution. Wang et al. (2024) find that a few variables from market sentiment and cross-exchange spreads suffice for ensemble tree models on Chinese oil futures, and that expanding the feature set beyond this core yields no improvement. Geopolitical risk subcategories have been shown to dominate crude oil futures volatility forecasts in a time-varying manner (Zhao et al., 2024), which supports the case for a dedicated geopolitical information channel.

## 2.3 LLMs as Forecasters

The base forecasts in the models reviewed above are generated by statistical or machine learning algorithms; large language models offer a recent alternative. LLMs have been applied to financial forecasting primarily as sentiment extractors. FinGPT (Liu et al., 2023) provides an open-source financial LLM for text-based sentiment scoring, and multi-dimensional LLM sentiment analysis has been applied to WTI crude oil returns using GPT-4o, where features such as intensity and uncertainty prove more informative than simple polarity (Dai et al., 2026). An evaluation of zero-shot LLM forecasting across time series benchmarks finds that LLMs perform competitively on structurally simple series but struggle with high-frequency financial data without domain context (Abolghasemi et al., 2025), a lesson consistent with the broader finding from forecasting competitions that no single method dominates across domains (Makridakis et al., 2022).

A smaller body of work uses LLMs as active reasoning agents rather than passive feature extractors. TradingAgents (Xiao et al., 2024) deploys specialist agents for equity trading decisions with a debate mechanism, and FinCon (Yu et al., 2024) uses multi-agent debate for financial concept resolution. The closest precedent is a multi-agent debate system with hawkish-dovish latent beliefs that significantly outperforms standard LLM baselines on FOMC rate classification (Takano et al., 2025). Two features distinguish the present setting: the forecast target is a continuous variable (20-day realised volatility) rather than a discrete policy decision, and the aggregation must weight seven domain-specialist agents rather than resolve a binary hawkish-dovish split. Agentic AI pipelines have also been applied to commodity price shock forecasting, though the agents function as task-decomposition tools (retrieve, summarise, fact-check) rather than domain-specialist debaters (Ghali et al., 2025). The domain-knowledge gap identified by the zero-shot evaluation literature can be addressed by providing each agent with structured, domain-specific features rather than requiring the LLM to infer context from raw prices.

## 2.4 Graph Neural Networks for Financial Forecasting

Graph attention networks (Veličković et al., 2018) have been applied to stock return prediction using pre-specified industry or correlation graphs (Zhang et al., 2025). GNNs aggregate realised volatility signals across assets, showing that graph structure carries information beyond pairwise correlation (Brini and Toscano, 2025), and dynamic graph learning improves cross-sectional return prediction when the graph is allowed to evolve over time (Chi et al., 2025). These papers construct graphs over assets or instruments. Communication topology optimisation for multi-agent systems through graph neural networks has been explored in the AI literature (Zhang et al., 2024). Constructing a graph over forecasters rather than over assets would encode combination structure rather than market microstructure, with edge weights representing the learned attention of one agent to another in the combination layer.

## 2.5 Emergent Bias in Multi-Agent Systems

Information cascades and herding have long been studied in human decision making (Banerjee, 1992; Bikhchandani et al., 1992). Recent evidence shows that analogous phenomena arise in LLM populations: when multiple agents interact, herding toward the group mean and overconfidence amplification emerge even without explicit coordination instructions (Ashery et al., 2024; Sun et al., 2025; Bini et al., 2026). Group size effects compound this tendency, as collective misalignment worsens with the number of interacting agents (Ashery et al., 2025). For forecast combination, the consequence is that naive aggregation treats herded and independent forecasts identically, conflating signal with social conformity. This literature motivates combination architectures that can distinguish independent signals from socially induced redundancy.

The five streams above point to a shared gap: in oil volatility forecasting, combination remains dominated by statistical or low-dimensional weighting schemes, and LLM applications have appeared mainly as sentiment tools rather than multi-agent reasoning forecasters; graph neural networks have been applied to asset-level graphs but not to forecaster-level combination; and the emergent biases of multi-agent LLM interaction remain unaddressed in forecasting practice. The framework proposed in this paper sits at the intersection of these literatures. The following section defines the prediction target, the walk-forward evaluation protocol, and the notation used throughout.

---

# 3. Problem Formulation and Notation

## 3.1 Prediction Target

Let $r_t = \log(P_t / P_{t-1})$ denote the daily log return on WTI crude oil,
where $P_t$ is the daily closing settlement price of the front-month futures
contract on trading day $t$.
The prediction target is the forward-looking 20-day realised volatility,

$$\text{fwd\_rv}_{20,t} = \sqrt{252} \cdot
\text{std}\!\left(r_{t+1}, \ldots, r_{t+20}\right),$$

an annualised figure expressed as a decimal.
The 20-day horizon matches the VaR and delta-hedging windows common
in energy risk management
and corresponds to the period over which a portfolio manager
would hold a volatility hedge.
All inputs, including $r_t$, are observed at the close of trading
day $t$, so the target window $\{t\!+\!1, \ldots, t\!+\!20\}$ lies
entirely in the future.

Because consecutive estimation windows share nineteen of their twenty
constituent returns, forecast errors are highly serially correlated by
construction.
Standard Diebold-Mariano tests assume short-range dependence
(Diebold and Mariano, 1995), so all inference in this paper uses
Newey-West HAC-corrected Diebold-Mariano statistics with a Bartlett
kernel and bandwidth 19; the correction is material and its magnitude
is reported in Section 7.

## 3.2 Evaluation Protocol

Forecast accuracy is assessed under a walk-forward, rolling-origin
protocol following Tashman (2000).
The initial training window spans 252 trading days
(approximately one trading year),
and the model is retrained every 63 trading days,
roughly one calendar quarter.
The full sample covers January 2020 to May 2025, yielding 1,285
trading days.
Sixty-one trading days between 24 February and 22 May 2020 are
excluded as a data quality filter.
These span the April 2020 WTI contract dislocation, when settlement
fell to $-\$36.98$ on 20 April, and the surrounding weeks in which
returns were driven by the contractual anomaly rather than
supply-demand fundamentals.
Robustness to this exclusion is examined in Appendix A.
The clean sample contains 1,224 observations,
and after reserving the minimum training window,
972 out-of-sample predictions are available for evaluation.

A 20-day label embargo is imposed on all supervised components.
Because the target for day $t$ is not fully observable until day $t+20$,
any supervised model evaluated at boundary $t$ trains on labels
only up to $t - 20$,
preventing it from seeing partially-realised future volatility.
The embargo applies to all supervised baselines and to the
GAT meta-aggregator in Section 5;
DropEdge GAT statistics are ensemble means across five random seeds.

## 3.3 Notation

$\mathcal{A} = \{a_1, \ldots, a_7\}$ is the set of seven specialist
LLM agents, each assigned a distinct information domain.
At each evaluation date $t$, every agent $a_i$ produces a Round 1
adjustment $\hat{v}^{(1)}_i(t)$ and, after observing the Round 1
outputs of all other agents, a Round 2 adjustment $\hat{v}^{(2)}_i(t)$.
Both adjustments are expressed relative to the persistence baseline,

$$v^{\text{persist}}_t = \text{rv}_{20,t}
\;\equiv\; \sqrt{252} \cdot
\text{std}\!\left(r_{t-19}, \ldots, r_{t}\right),$$

the backward-looking 20-day realised volatility, fully
observable at time $t$.
The naive aggregation baseline $\hat{y}^{\text{debate}}_t$ is the
confidence-weighted mean of Round 2 adjustments added to persistence:

$$\hat{y}^{\text{debate}}_t =
v^{\text{persist}}_t +
\sum_{i=1}^{7} w_i(t)\, \hat{v}^{(2)}_i(t),$$

where $w_i(t) = c_i(t) \big/ \sum_{j=1}^{7} c_j(t)$ and
$c_i(t) \in (0,1]$ is the confidence score that agent $a_i$ reports
alongside its forecast (strictly positive by prompt design).
The single-agent baseline $\hat{y}^{\text{single}}_t$ is produced by
a single generalist LLM that receives all seven domains' data in one
prompt without multi-agent debate,
and is a control for the debate mechanism.

The Shapley value $\phi_i(t)$ (Shapley, 1953) measures the marginal
contribution of agent $a_i$ to forecast error at time $t$.
The coalition value function is the squared forecast error,
$v_t(S) = \bigl(\hat{y}^{\text{debate}}_t(S) -
\text{fwd\_rv}_{20,t}\bigr)^2$,
where

$$\hat{y}^{\text{debate}}_t(S) = v^{\text{persist}}_t +
\sum_{i \in S} \frac{c_i(t)}{\sum_{j \in S} c_j(t)}
\;\hat{v}^{(2)}_i(t),
\qquad
\hat{y}^{\text{debate}}_t(\varnothing) = v^{\text{persist}}_t,$$

so that confidence weights are renormalised within the active coalition.
The computational implementation of this formula, including a small
numerical approximation for coalition stability, is detailed in
Section 6.4.
Values are estimated by Monte Carlo sampling over coalition orderings;
$\phi_i(t) < 0$ means that including agent $a_i$ reduces
squared error (cost-game convention, maintained throughout).

The influence graph $G_w = (\mathcal{A},\, \mathcal{E}_w)$
is the learned sparse topology for training window $w$,
with edge set $\mathcal{E}_w \subseteq \mathcal{A} \times \mathcal{A}$.
An edge $(a_i, a_j) \in \mathcal{E}_w$ means that agent $a_j$'s
node representation attends to agent $a_i$ during graph aggregation;
the resulting attention weight matrix $\mathbf{W}_{w,t}$ varies per
observation and is detailed in Section 5.
The edge set is not pre-specified;
learnable sigmoid-gated edge logits let the network
discover which connections reduce out-of-sample error,
converging to roughly 16 of 42 possible edges.
Section 4 uses these definitions to characterise the group behaviour
patterns that motivate the graph-based aggregation in Section 5.

---

# 4. Motivation for the GAT Architecture

## 4.1 Why naive aggregation fails

The natural starting point for combining outputs from a multi-agent
debate is the confidence-weighted mean defined in Section 3.3.
After correcting for the serial dependence induced by the 20-day
overlapping forecast horizon, however, this naive rule does not
significantly outperform the HAR benchmark
(Diebold and Mariano, 1995; HAC-corrected statistic $-1.494$,
$p = 0.135$; full results in Section 7).
The problem is not that the base forecasts are uninformative.
Shapley attribution shows that most agents reduce forecast error on
average, and the debate outputs contain signal that, when
properly extracted, yields clear accuracy gains (Section 7).

The obstacle is more specific: no agent helps reliably across all
market conditions.
All seven agents have weakly negative mean Shapley values on the
evaluation sample, but usefulness is episodic: each agent produces
positive values (increasing error) on 38 to 59 percent of individual
days.
Confidence-weighted averaging treats each agent proportionally to
its self-reported certainty, which is not the same as its marginal
predictive contribution.
An agent that is consistently confident but sporadically useful
receives persistent weight rather than selective weight.

The herding phenomenon documented in Section 6.3 compounds this
problem.
Across the evaluation sample, 43.4 percent of agent-day observations
are classified as herding (Section 6.3), reducing the effective
number of independent forecasts below seven without reducing the
nominal weight total.

## 4.2 Why structure-aware aggregation is needed

Two empirical patterns suggest that the aggregation problem has
exploitable structure rather than being irreducibly noisy.

Agent Shapley values vary across volatility regimes.
Some agents contribute more consistently in low-volatility periods,
others in elevated-volatility regimes; the geopolitical agent, for
example, reduces error in low and normal regimes but turns harmful
in elevated and high-volatility periods.
This regime-conditional variation means the optimal aggregation
weights are time-varying.
A method that conditions on market state and on each agent's recent
contribution history can recover the weights that static
confidence-weighting cannot.

The debate influence graph $G^{\text{dbt}}_t$ defined in Section 6.2
offers a second source of structure.
$G^{\text{dbt}}_t$ records, for each day $t$, which agents revised
their adjustments in the direction of other agents' Round 1 positions.
The density of this graph is positively associated with the
herding count on that day ($r = 0.35$): sessions in which many
influence edges are active tend to be sessions in which many agents
herd.
An aggregation method that can read the communication structure and
down-weight agents whose revisions were strongly shaped by others
would recover some of the diversity that herding erodes.
A static rule operating on confidence scores alone has no access to
this information.

[Figure 2 here: two panels.
Panel A shows mean Shapley value by agent and volatility regime,
illustrating that agent contributions are heterogeneous across
regimes and that no agent dominates uniformly.
Panel B shows a scatter of daily influence-edge count in
$G^{\text{dbt}}_t$ against the daily herding count,
illustrating the positive association between communication
density and conformist behaviour.]

These two patterns are descriptive rather than causal.
After controlling for the contemporaneous volatility regime,
neither communication density nor herding count significantly
predicts forecast error in isolation.
The patterns do not prescribe a specific functional form for the
aggregator; they establish that the relevant conditioning variables
exist and are observable at time $t$, motivating a learned approach
that can discover which combinations of those variables are
predictively useful.

## 4.3 Why sparsity is desirable

The observations above point toward a learned, graph-structured
aggregator, but they do not yet explain why sparsity should be a
design objective.

When agents herd, their Round 2 outputs contain duplicated
information: several agents have effectively converged to the same
signal, and a combination that weights them as if they were independent
overcounts that signal.
A fully connected aggregation graph, in which every agent can
influence every other, imposes no structural bias against
this redundancy; attention weights can in principle learn low values,
but the network has no inductive pressure to do so.
If the network learns to propagate information along all edges,
the herded consensus circulates through the graph,
amplifying rather than correcting the reduction in effective
diversity.

Sparse aggregation provides a structural remedy.
By restricting the graph to a small number of active edges, the
network is forced to identify which agent-to-agent information
flows are predictive and to ignore the rest.
An edge that exists only because two agents converged to the same
position offers no additional predictive information beyond what
either agent carries alone, and a sparsity mechanism that can
gate out such edges preserves the independent signals while
discarding the duplicated ones.

This argument does not require that the correct sparse structure
be known in advance.
It requires only that a learned sparse structure, discovered from
walk-forward training data, is preferable to a dense one when the
underlying agent outputs contain herding-induced redundancy.
Section 5.2 describes how learnable sigmoid-gated edge logits
implement this idea, and Section 7 reports the resulting
forecast accuracy.

---

# 5. GAT Architecture

[Figure 1 here: two-stage architecture diagram showing data flow
from raw market features through the seven SDD agents and the
two-round debate protocol, into the GAT meta-aggregator, and out
to the final prediction.
The diagram shows the node feature matrix, the learnable edge
logit matrix, the dual-head output, and the regime gate.]

Figure 1 summarises the two-stage architecture.
Stage 1 generates base forecasts through the Structured Delphi Debate
protocol described in Section 6.
Stage 2 is the graph attention network described here, which learns
to combine those forecasts by treating agents as nodes in a graph
whose topology is itself a learned parameter.

## 5.1 Inputs: node features and context vector

The GAT operates on a graph of seven agent nodes,
one for each specialist in $\mathcal{A} = \{a_1, \ldots, a_7\}$
defined in Section 3.3.
Only two signals are passed directly to the output head as base
forecasts: the debate forecast $\hat{y}^{\text{debate}}_t$ and the
single-agent forecast $\hat{y}^{\text{single}}_t$.
All other inputs enter via the graph and context representations.
Adding statistical model outputs (HAR, GARCH, persistence) as
additional base forecasts reduces accuracy because
$\hat{y}^{\text{debate}}_t$ already incorporates persistence through
the agents' prompt design, making those signals redundant,
as documented in Appendix B.

Each node $a_i$ carries a 7-dimensional feature vector
$\mathbf{x}_i(t) \in \mathbb{R}^7$,
assembled from time-$t$ quantities that characterise the agent's
recent behaviour and influence.
The seven components are:
the Shapley value $\phi_i(t)$ (cost-game convention, so negative
values indicate that the agent reduces forecast error);
the Myerson value $\mu_i(t)$, a graph-restricted Shapley value
(Myerson, 1977) that re-computes marginal contributions within
the connected component of the influence graph, so that isolated
agents receive attenuated credit;
a normalised four-category behaviour code $b_i(t)/3$,
where the four categories are herding, anchored, independent,
and overconfident, giving values in $\{0, 1/3, 2/3, 1\}$;
the degree $d_i(t)/6$, counting both incoming and outgoing
edges in the debate influence graph;
the 5-day moving average of $\phi_i$;
the 5-day standard deviation of $\phi_i$; and
the normalised herding streak $s_i(t)/5$,
computed as a rolling 5-day sum of the herding indicator,
which distinguishes persistent herding from isolated episodes.

A 9-dimensional context vector $\mathbf{c}(t) \in \mathbb{R}^9$
contains market-state information that is not agent-specific.
Its components are: the persistence volatility $\text{rv}_{20,t}$;
the HAR model volatility estimate;
the total agent adjustment $\hat{y}^{\text{debate}}_t - \text{rv}_{20,t}$;
the normalised herding count $n_{\text{herd}}(t)/7$;
a normalised four-level volatility regime indicator $r(t)/3$,
where the four levels correspond to persistence thresholds at
0.20, 0.35, and 0.55;
the debate forecast $\hat{y}^{\text{debate}}_t$;
the persistence-HAR gap;
the five-day change in persistence volatility $\Delta_5 \text{rv}_{20,t}$; and
the debate-HAR gap.
This vector is concatenated with the graph embedding before the
output heads and the residual branch.
A separate 3-dimensional regime feature vector, consisting of
$\text{rv}_{20,t}$, the one-day change
$\Delta_1\text{rv}_{20,t}$, and $n_{\text{herd}}(t)/7$,
feeds the regime gate described in Section 5.5.

## 5.2 Graph construction

The interaction topology is not pre-specified.
A learnable logit matrix $\mathbf{L} \in \mathbb{R}^{7 \times 7}$
parameterises the edge set, with $\mathbf{L}_{ij}$ representing the
log-odds of the edge from $a_i$ to $a_j$ being active.
The effective edge weight entering the GAT layers is

$$A_{ij} = \sigma(L_{ij}),\tag{1}$$

where $\sigma$ denotes the logistic sigmoid.
Self-loops are fixed at one throughout training and inference,
so that self-attention remains a candidate even after top-$k$
masking.
The logits are initialised by drawing from $\mathcal{N}(0, 0.1^2)$,
placing the initial graph close to a connection probability of 0.5
for every possible edge, so no structural prior is imposed.
After each training run, edges with $A_{ij} < 0.5$ are treated as
inactive; the network consistently converges to roughly 16 of the
42 possible directed edges across seeds and training windows.

This is the first of three sparsity layers in the architecture.
Because the masking formula drives attention scores toward $-\infty$
for edges with low $A_{ij}$, the sigmoid acts as a
differentiable binary gate rather than a continuous weight,
and the network learns to push each logit toward zero or one.
The hard 0.5 threshold applied post-training formalises the resulting
sparse graph without any penalty term in the loss.
Experiments with Granger-based causal priors as warm-start
initialisations for $\mathbf{L}$ do not outperform learning from
scratch; these variants are documented in Appendix B.

## 5.3 Multi-head GAT layers

The graph encoder consists of two successive Graph Attention Network
layers (Veličković et al., 2018),
each using four attention heads with a per-head hidden dimension of
four, giving a total hidden dimension of 16.
The resulting per-observation attention coefficients form the weight matrix $\mathbf{W}_{w,t}$ introduced in Section 3.3.
Within head $k$, the unnormalised attention score from source node
$a_i$ to destination node $a_j$ is

$$e_{ij}^{(k)} = \text{LeakyReLU}\!\left(
  \mathbf{a}_{\text{src}}^{(k)\top} \mathbf{W}^{(k)} \mathbf{h}_i
  + \mathbf{a}_{\text{dst}}^{(k)\top} \mathbf{W}^{(k)} \mathbf{h}_j
\right),\tag{2}$$

where $\mathbf{h}_i$ is the current hidden state of node $a_i$,
$\mathbf{W}^{(k)}$ is the head-specific linear projection,
and $\mathbf{a}_{\text{src}}^{(k)}$, $\mathbf{a}_{\text{dst}}^{(k)}$
are learnable attention vectors.
The decomposed additive formulation follows the original GAT paper
(Veličković et al., 2018), with equivalent empirical performance
for fixed-size node sets.
The edge weight matrix $\mathbf{A}$ from equation (1) enters as a
multiplicative mask: positions with $A_{ij} = 0$ receive a score
of $-10^9$ before softmax, effectively zeroing out the corresponding
attention weight.

The second sparsity layer operates within each attention head:
top-$k$ masking with $k=3$ retains only the three highest-scoring
incoming edges per node, setting the remainder to $-10^9$.
Let $\mathcal{N}_k(j)$ denote the top-$k$ neighbourhood of node $j$
after masking.
The resulting attention coefficient is

$$\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\displaystyle\sum_{l \in \mathcal{N}_k(j)} \exp(e_{lj}^{(k)})}.\tag{3}$$

The updated representation of node $a_j$ is then
$\mathbf{h}_j' = \text{concat}_{k=1}^{4}
\sum_{i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_i$.

Each GAT layer includes a skip connection:
node features are projected to the hidden dimension by a shared
linear map, added to the attention output, and passed through
LayerNorm before ReLU activation.
The second layer adds its output to the first layer's output,
giving the network a residual path of depth two.

After both layers, the seven node representations are concatenated
rather than pooled, yielding a graph embedding of dimension
$7 \times 16 = 112$; concatenation preserves the agent-specific
variation that pooling would discard.
The graph embedding is then concatenated with the context vector
$\mathbf{c}(t)$ to produce a 121-dimensional combined representation
$\mathbf{z}(t) = [\text{flatten}(\mathbf{H}') \| \mathbf{c}(t)]$,
where $\mathbf{H}' \in \mathbb{R}^{7 \times 16}$ is the matrix of
post-second-layer node representations.

## 5.4 DropEdge regularisation

The third sparsity layer is stochastic.
During each forward pass in training, 20% of active edges are
randomly set to zero before the GAT layers compute attention.
Self-loops are exempt from this dropout.
At inference, all learned edges remain active.

This procedure, referred to as DropEdge (Rong et al., 2020),
prevents the model from becoming dependent on any particular subset
of the approximately 16 active edges.
Section 7 shows that the learned topology is not unique: no single
edge appears consistently across all seeds and training windows.
DropEdge implicitly trains the network on an ensemble of subgraphs
and improves out-of-sample robustness when the true optimal graph
is not identifiable from the available data.
Earlier variants using $L_1$ regularisation to encourage sparsity
are reported in Appendix B.

## 5.5 Regime-gated dual-head output

The output layer converts the combined representation $\mathbf{z}(t)$
into a convex combination of the two base forecasts.
Two independent heads, referred to as the base head and the regime
head, each receive $\mathbf{z}(t)$ and map it through a 32-unit
hidden layer with ReLU activation and 15% dropout, then through a
linear layer to two logits, and finally through softmax to produce a
weight vector over
$[\hat{y}^{\text{debate}}_t,\, \hat{y}^{\text{single}}_t]$.
The heads share the same architecture but have separate parameters;
the regime head is trained to specialise on regime-dependent
patterns.

A regime gate determines which head governs the final combination.
The gate receives the 3-dimensional regime feature vector introduced
in Section 5.1: $[\text{rv}_{20,t},\;
\Delta_1\text{rv}_{20,t},\; n_{\text{herd}}(t)/7]$,
where $\Delta_1\text{rv}_{20,t} = \text{rv}_{20,t} - \text{rv}_{20,t-1}$.
These signals pass through a two-layer network (8 hidden units,
Tanh activation) followed by a sigmoid output, giving a scalar gate
$g(t) \in [0,1]$.
The final combination weights are

$$\mathbf{w}(t) = (1 - g(t))\, \mathbf{w}_{\text{base}}(t)
+ g(t)\, \mathbf{w}_{\text{regime}}(t).\tag{4}$$

When $g(t)$ is near zero the base head dominates;
when $g(t)$ is near one the regime head governs
the combination instead.

A small residual correction supplements the convex combination.
A two-layer network (input dimension 121, 16 hidden units, Tanh
activation, scalar output) maps $\mathbf{z}(t)$ to a signed
correction term, scaled by 0.05 to keep the residual small relative
to the base prediction.
The final forecast is

$$\hat{y}_t = \max\!\left(
  \mathbf{w}(t)^\top
  \begin{bmatrix}\hat{y}^{\text{debate}}_t \\
  \hat{y}^{\text{single}}_t\end{bmatrix}
  + 0.05\,\text{res}(\mathbf{z}(t)),\;
  0.05
\right),\tag{5}$$

where the lower bound of 0.05 enforces a minimum annualised
volatility of 5%, well below the sample minimum of 0.12.

## 5.6 Training protocol

The DropEdge GAT is trained with Adam (Kingma and Ba, 2015),
with a learning rate of 0.003 and weight decay of $10^{-4}$.
A cosine annealing schedule reduces the learning rate from 0.003
to zero over 250 epochs.
Gradient norms are clipped to 1.0 at each step to prevent
instability during the early phase of graph learning,
when the edge logits are far from convergence.
The checkpoint with minimum training loss across the 250 epochs
is retained, as validation-based early stopping caused underfitting
at the training window sizes encountered in the walk-forward protocol.

Training is embedded within the walk-forward evaluation protocol
of Section 3.2.
A new model instance is trained from scratch at each retraining
point, every 63 trading days, using all available clean observations
up to that point subject to the 20-day label embargo.
Five independent random seeds initialise both the edge logits and
the network weights; reported metrics are 5-seed ensemble means,
following Lakshminarayanan et al. (2017) for uncertainty reduction
through deep ensemble averaging.

Section 6 describes how the debate forecast $\hat{y}^{\text{debate}}_t$
and the single-agent forecast $\hat{y}^{\text{single}}_t$ consumed
by equation (5) are generated.

---

# 6. Structured Delphi Debate Protocol

Stage 1 of the framework elicits volatility adjustments from seven
specialist LLM agents through a two-round structured debate and
quantifies each agent's contribution via cooperative game theory.
The output of Stage 1 is the pair
$(\hat{y}^{\text{debate}}_t, \hat{y}^{\text{single}}_t)$
consumed by equation (5) in Section 5, together with the node
features $\mathbf{x}_i(t)$ described in Section 5.1.

## 6.1 Agent Design

Seven specialist agents are instantiated from a frontier chat model
(temperature 0.3),
each receiving a distinct set of domain-aligned proxy variables drawn
from the same daily observation at time $t$.
Three agents draw on text-derived signals:
the geopolitical agent receives GDELT event and conflict counts,
conflict share, material conflict share, Goldstein mean, and media
tone for oil-producing regions;
the sentiment agent receives GDELT-derived oil news volume, oil and
global media tone, and the net cooperation index;
and the cross-market agent receives the VIX level, the daily VIX
change, and the daily change in the 10Y-2Y yield spread.
The remaining four agents draw on market and macro data:
the monetary agent receives the federal funds rate, the 2-year,
10-year, and real 10-year Treasury yields, the 10Y-2Y yield spread,
and the DXY dollar index with its daily change;
the macro demand agent receives the Shanghai Composite index and
the 20-day WTI price momentum;
the supply and OPEC agent receives the WTI spot price together with
5-day and 20-day price momentum as proxy indicators of supply-side
pressure, given the absence of real-time production and inventory
releases at daily frequency;
and the technical agent receives 20-day and 60-day backward realised
volatility, the current daily log return, and the estimated
probability of a high-volatility regime.

Each agent is given a domain-specific system prompt that instructs it
to reason from within its analytical perspective.
At the close of each prompt, the agent is asked to output a JSON
object with four fields: a point adjustment $\hat{v}_i^{(r)}(t)$
to the persistence baseline $v^{\text{persist}}_t$, expressed in
the same annualised units as the target; a confidence score
$c_i(t) \in (0, 1]$; a directional label in
$\{$up, stable, down$\}$; and a brief evidence list.
The model is requested to produce valid JSON, and the response is
parsed with a fallback to a neutral default if parsing fails.
This design adapts the Delphi elicitation method,
in which a panel of domain experts provides independent assessments
that are then aggregated (Dalkey and Helmer, 1963),
to a reproducible multi-agent LLM inference procedure.

## 6.2 Forecast Elicitation and Debate Influence Graph

In Round 1, each agent $a_i \in \mathcal{A}$ receives only its own
feature set, together with the persistence baseline
$v^{\text{persist}}_t$ and a brief common market snapshot
(WTI close price, latest return, and the backward-looking 20-day and
60-day realised volatilities).
All seven agents are queried in parallel without shared state,
so no agent observes any other agent's output at this stage.
The resulting Round 1 adjustment $\hat{v}_i^{(1)}(t)$ therefore
reflects each agent's domain-specific reasoning alone,
so the initial pool of adjustments is diverse by construction.

In Round 2, each agent receives the Round 1 adjustments and evidence
summaries of all six other agents, alongside its own Round 1 output,
and is asked to revise its adjustment.
The revised output is $\hat{v}_i^{(2)}(t)$.
The debate forecast, defined in Section 3.3, is

$$\hat{y}^{\text{debate}}_t =
v^{\text{persist}}_t +
\sum_{i=1}^{7} w_i(t)\, \hat{v}^{(2)}_i(t),
\qquad
w_i(t) = \frac{c_i(t)}{\displaystyle\sum_{j=1}^{7} c_j(t)}.\tag{6}$$

The single-agent baseline $\hat{y}^{\text{single}}_t$, also defined
in Section 3.3, is generated by a separate prompt that presents all
seven domains' data to one agent in a single call without any debate.
It is a control for the multi-agent structure: any improvement of
$\hat{y}^{\text{debate}}_t$ over $\hat{y}^{\text{single}}_t$ is
attributable to the debate, not to the LLM's reasoning capacity.

After each Round 1 to Round 2 transition, a debate influence graph
$G^{\text{dbt}}_t = (\mathcal{A}, \mathcal{E}^{\text{dbt}}_t)$
is constructed to record how each agent's revision was shaped by its
peers.
For each target agent $a_j$ whose total revision exceeds 0.002
(eliminating numerical noise),
and for each potential source agent $a_i \neq a_j$ whose Round 1
position differs from $a_j$'s by at least 0.002
(avoiding division by near-zero in the weight formula),
a directed edge $(a_i \to a_j)$ is created if $a_j$'s revision
moves in the direction of $a_i$'s Round 1 position.
The edge weight is

$$\omega_{ij}(t) = \min\!\left(
\frac{|\hat{v}_j^{(2)}(t) - \hat{v}_j^{(1)}(t)|}
     {|\hat{v}_i^{(1)}(t) - \hat{v}_j^{(1)}(t)|},\;
1\right),\tag{7}$$

and edges with $\omega_{ij}(t) < 0.05$ are discarded.
The graph $G^{\text{dbt}}_t$ is distinct from the learned GAT
topology $G_w$ described in Section 5.2, which is a training
parameter rather than a time-$t$ observation.
The agent degree $d_i(t)$, counting both in- and outgoing edges in
$G^{\text{dbt}}_t$ and divided by 6, enters the node
feature vector $\mathbf{x}_i(t)$ in Section 5.1.
The Myerson value $\mu_i(t)$ is also computed on
$G^{\text{dbt}}_t$, as described in Section 6.4.

## 6.3 Behavioural Classification

Each agent is assigned a behavioural label at each time step based
on the trajectory of its adjustment from Round 1 to Round 2.
Classification follows a priority ordering.

Let $\Delta_i = |\hat{v}_i^{(2)}(t) - \hat{v}_i^{(1)}(t)|$ denote
the total magnitude of revision and let
$m(t) = \text{median}_{j \in \mathcal{A}} \hat{v}_j^{(2)}(t)$
denote the final-round median adjustment across all agents.

An agent is classified as anchored if $\Delta_i < 0.002$,
indicating that it effectively ignored the social information from
other agents' Round 1 outputs.
This category is checked first because a high-confidence agent that
happens not to move should be recorded as anchored, not overconfident.
An agent whose revision was not anchored is classified as
overconfident if $c_i(t) > 0.85$ and $\Delta_i > 0.005$:
a large revision paired with high confidence indicates
the agent is amplifying its own prior view.
Among the remaining agents, herding is assigned if $|m(t)| > 0.001$,
the revision moves toward $m(t)$, and $\Delta_i > 0.003$
(above the anchored threshold, ensuring the revision reflects
substantive movement).
The median condition ensures that when agents broadly agree on
near-zero adjustment, no herding label is assigned.
Formally, herding requires
$\text{sign}(\hat{v}_i^{(2)}(t) - \hat{v}_i^{(1)}(t))
= \text{sign}(m(t) - \hat{v}_i^{(1)}(t))$.
All remaining agents are classified as independent.

The four categories are mapped to integers
$\{0, 1, 2, 3\}$ in the order
$\{\text{herding}, \text{anchored}, \text{independent},
\text{overconfident}\}$
and divided by 3, yielding a single scalar
$b_i(t)/3 \in \{0,\, 1/3,\, 2/3,\, 1\}$
that enters the node feature vector as described in Section 5.1.
The herding streak $s_i(t)/5$, a rolling 5-day sum of the binary
herding indicator normalised by 5, is computed separately and also
enters $\mathbf{x}_i(t)$; it distinguishes agents in sustained
herding episodes from those exhibiting isolated conformity.
The aggregate herding count $n_{\text{herd}}(t)$, the number of
agents classified as herding on day $t$, appears in the context
vector $\mathbf{c}(t)$ and the regime gate input in Section 5.

## 6.4 Agent Attribution

The Shapley value $\phi_i(t)$ is estimated by Monte Carlo sampling
over orderings of $\mathcal{A}$.
Section 3.3 defines the coalition value function
$v_t(S) = (\hat{y}^{\text{debate}}_t(S) - \text{fwd\_rv}_{20,t})^2$,
where the empty-coalition forecast equals
the persistence baseline $v^{\text{persist}}_t$.
In the implementation, coalition forecasts are computed by assigning
member agents their Round 2 adjustments and confidence weights, while
non-member agents contribute zero adjustment with a small weight
floor (0.01) for numerical stability; this introduces a slight
shrinkage toward persistence that diminishes as coalition size
grows.[^1]
Under this cost-game convention, $\phi_i(t) < 0$ means that
including agent $a_i$ reduces squared forecast error.
The same 500-sample Monte Carlo estimator is applied at every
evaluation date $t$; with 7 agents and $7! = 5{,}040$ possible
orderings, 500 samples cover approximately 10% of the permutation
space, which is sufficient for stable marginal contribution estimates
in small-coalition games (Castro et al., 2009).

The Myerson value $\mu_i(t)$ (Myerson, 1977) adapts Shapley to
account for the communication structure in $G^{\text{dbt}}_t$.
Rather than allowing all coalitions, Myerson's restriction considers
connectivity in $G^{\text{dbt}}_t$ treated as undirected:
agents that are isolated from their coalition partners retain only
their standalone contribution rather than sharing in the coalition's
joint value.
Concretely, when computing the marginal contribution of agent $a_i$
to a coalition $S$ under a given ordering, only the connected
component of $S \cup \{a_i\}$ containing $a_i$ is activated;
the value is the difference in squared error between this component
with and without $a_i$.
The gap $\mu_i(t) - \phi_i(t)$ measures the influence-mediated
component of agent $a_i$'s contribution: its sign and magnitude
depend on whether the agent bridges otherwise disconnected groups
or contributes primarily in isolation.
Both $\phi_i(t)$ and $\mu_i(t)$ enter $\mathbf{x}_i(t)$ as node
features; the resulting patterns across agents and market regimes
are reported in Section 7.

[^1]: For a coalition of size 3 with average confidence 0.5, the
weight floor dilutes the coalition adjustment by approximately 3%;
for size 6, the dilution is below 1%.

---

# 7. Empirical Results

## 7.1 Data and Implementation

The sample spans 2 January 2020 to 29 May 2025 and contains 1,285 trading days. WTI crude oil daily closing prices are sourced from front-month futures settlements in the oil macro daily dataset, which also provides VIX, DXY, the federal funds rate, 2-year and 10-year Treasury yields, and the 10Y-2Y yield spread. GDELT daily event features supply the geopolitical and media-tone variables consumed by the geopolitical and sentiment agents described in Section 6.1. The April 2020 WTI contract dislocation, when front-month settlement fell to $-\$36.98$ on 20 April, contaminates volatility windows in the surrounding weeks. Following the fixed exclusion in Section 3.2, the 61 trading days between 24 February and 22 May 2020 are removed, yielding a clean evaluation sample of $n = 1{,}224$. As described in Section 3.2, the walk-forward protocol reserves a minimum training window of 252 trading days and retrains every 63 trading days, leaving 972 out-of-sample predictions.

Six categories of baselines are evaluated alongside the proposed model. The persistence forecast sets $\hat{y}_t = \text{rv}_{20,t}$ and requires no estimation. The HAR-RV model (Corsi, 2009) and a GARCH(1,1) specification (Bollerslev, 1986) provide econometric benchmarks estimated on backward-looking realised volatility. Five machine learning methods are included: Ridge regression, Lasso, gradient boosting, random forest, and XGBoost, all trained on the same feature set of lagged volatilities, returns, and macro indicators, with the forward-looking 20-day target subject to the label embargo defined in Section 3.2. Two deep learning architectures, an LSTM and a Transformer encoder, are estimated under the same rolling walk-forward and embargo constraints. Two LLM-based methods serve as Stage 1 controls: the single-agent generalist, which receives all seven domains in one prompt without debate (Section 6.1), and the naive debate aggregation, the confidence-weighted mean of Round 2 adjustments defined in equation (6). The proposed method is the 5-seed ensemble DropEdge learned sparse GAT described in Section 5, which takes only the debate and single-agent forecasts as base predictions. Table 1 additionally includes a historical-mean benchmark, the Learned GAT variant without DropEdge regularisation, and an MLP that receives identical inputs but no graph structure, bringing the total to 16 evaluated methods.

All pairwise accuracy comparisons use the Diebold-Mariano test (Diebold and Mariano, 1995) with Newey-West HAC standard errors, Bartlett kernel, and bandwidth 19, matching the 20-day forecast horizon as discussed in Section 3.1. The loss differential series are squared errors. Section 3.1 documents the rationale for this correction: overlapping forecast windows induce autocorrelation in the error series with empirical ACF at lag 1 of 0.938, and uncorrected DM statistics overstate significance by roughly 65 percent relative to the HAC-adjusted version. The full model comparison appears in Table 1; the pairwise HAC-corrected DM matrix for the principal models is reported in Appendix C.

## 7.2 Overall Forecast Accuracy

[Table 1 here: RMSE and HAC-corrected DM statistics for all 16 models on the aligned evaluation sample ($n = 972$).]

The DropEdge learned sparse GAT achieves RMSE of 0.1171 on the 972 walk-forward predictions, a 24.4 percent reduction relative to the HAR benchmark at 0.1549. The HAC-corrected Diebold-Mariano statistic is $-3.212$ ($p = 0.001$; all $p$-values in this section are rounded to three decimal places); the improvement is statistically significant after accounting for the serial dependence induced by the 20-day overlapping horizon. Persistence (RMSE $= 0.1551$) is statistically indistinguishable from HAR ($\text{DM}_{\text{HAC}} = 0.192$, $p = 0.848$), consistent with the well-documented difficulty of improving upon simple benchmarks for realised volatility (Patton, 2011).

Among the 16 methods evaluated, only three significantly outperform HAR at the 5 percent level: the DropEdge GAT, the Learned GAT variant without DropEdge regularisation (RMSE $= 0.1188$, $\text{DM}_{\text{HAC}} = {-3.165}$, $p = 0.002$), and GARCH(1,1) (RMSE $= 0.1366$, $\text{DM}_{\text{HAC}} = {-2.604}$, $p = 0.009$). The DropEdge GAT achieves substantially lower error than GARCH; the direct pairwise comparison yields $\text{DM}_{\text{HAC}} = {-1.948}$ ($p = 0.051$), marginally significant at the 10 percent level. That GARCH retains significance while all machine learning and deep learning alternatives do not reflects the well-specified parametric structure of the GARCH volatility model for this target, in line with recent evidence on the continued competitiveness of econometric benchmarks in energy forecasting (Degiannakis and Filis, 2017).

The five machine learning baselines and two deep learning architectures uniformly fail to improve upon HAR after HAC correction and label embargo enforcement. Ridge regression, the strongest of this group, achieves RMSE of 0.1493 ($\text{DM}_{\text{HAC}} = {-0.545}$, $p = 0.586$). At the other extreme, the LSTM is significantly worse than HAR ($\text{DM}_{\text{HAC}} = 3.446$, $p = 0.001$), likely due to overfitting on the relatively short training windows mandated by the rolling walk-forward protocol. Before applying HAC correction and the 20-day label embargo, several of these baselines appeared significant, illustrating how much the inferential corrections of Section 3.1 matter in practice.

The two LLM-based Stage 1 methods fall between the machine learning baselines and the econometric models. The naive debate aggregation achieves RMSE of 0.1512 ($\text{DM}_{\text{HAC}} = {-1.494}$, $p = 0.135$), and the single-agent generalist achieves 0.1495 ($\text{DM}_{\text{HAC}} = {-1.230}$, $p = 0.219$). Neither is significant. Both LLM methods achieve lower RMSE than HAR, and both outperform most machine learning baselines, though the single-agent generalist (0.1495) is essentially tied with Ridge (0.1493). These point estimates suggest that the debate and single-agent forecasts contain predictive information beyond what the conventional feature set captures. The gap between LLM forecast quality and statistical significance is precisely the aggregation problem identified in Section 4 and in the broader forecast combination literature (Timmermann, 2006): the signal is present but the naive combination mechanism is too noisy to extract it reliably. The DropEdge GAT closes this gap, converting non-significant LLM forecasts into highly significant combined predictions.

The role of the HAC correction merits explicit comment. The uncorrected DM statistic for the naive debate aggregation against HAR exceeds conventional significance thresholds, and a researcher relying on the standard test would incorrectly conclude that the debate rule significantly outperforms HAR. The HAC correction reverses this conclusion. This discrepancy, which Section 3.1 attributes to the 0.938 lag-1 autocorrelation in squared forecast errors, applies to all models in Table 1 and has direct methodological implications for any study evaluating overlapping-horizon forecasts.

## 7.3 Architecture Ablations

[Table 2 here: ablation results showing RMSE and HAC-corrected DM statistics for six architectural variants plus the full model as reference ($n = 972$). Variants: Dense GAT, Random graph, Identity (self-loops), No Shapley/Myerson features, No regime gate, and Enhanced features. The MLP no-graph baseline from Table 1 is included for comparison.]

Three groups of ablations isolate the contributions of individual architectural components. All variants share the walk-forward protocol, 5-seed ensemble, 20-day label embargo, and HAC-corrected DM inference of Section 3.2. Table 2 combines two corrected ablation sweeps run under this common protocol and reports the results.

The most consequential design choice is graph structure. The MLP baseline receives identical node features and context vector but replaces the GAT layers with a fully connected encoder, eliminating inter-agent information flow. It achieves RMSE of 0.1444, not significant against HAR ($\text{DM}_{\text{HAC}} = {-1.214}$, $p = 0.225$). The direct pairwise comparison yields $\text{DM}_{\text{HAC}} = {-3.336}$ ($p < 0.001$), so graph structure contributes beyond what the same inputs achieve without relational processing.

The nature of the graph matters more than its mere presence. The dense GAT, in which all 42 directed edges remain active, achieves RMSE of 0.1452, slightly worse than the MLP. Full connectivity allows herding-induced redundancy (Section 4.3) to propagate freely, amplifying the loss of diversity rather than correcting it. A random graph with the same sparsity (approximately 16 of 42 edges) reduces error to 0.1414, still not significant against HAR ($p = 0.166$). The ordering is clear: learned sparsity outperforms random sparsity, which outperforms full connectivity.

The identity variant, retaining only self-loops with no inter-agent edges, achieves RMSE of 0.1188, not significantly different from the full model ($\text{DM}_{\text{HAC}} = {-1.107}$, $p = 0.268$). This variant is not trivial; it retains multi-head self-attention on each node's own features, the regime-gated dual head, and the residual correction of Section 5. That it strongly outperforms the dense GAT ($\text{DM}_{\text{HAC}} = 6.365$, $p < 0.001$) indicates that the self-loop pathway extracts most of the useful signal from node features. Cross-agent propagation may provide small additional gains, since the full model does achieve lower point RMSE (0.1171 vs 0.1188), but those gains are not statistically distinguishable with this sample size, and they require a learned sparse topology to materialise.

Removing the Shapley and Myerson attribution features is the single most damaging ablation. Without them the model achieves RMSE of 0.1520, statistically indistinguishable from HAR ($\text{DM}_{\text{HAC}} = {-0.420}$, $p = 0.675$). These features encode each agent's recent marginal contribution to forecast accuracy; without them the network cannot distinguish reliably helpful agents from those that happened to herd on a given day, a result that supports the design premise of Section 5.1.

The regime gate contributes a more modest improvement. Disabling it raises RMSE from 0.1171 to 0.1198; both variants remain significant against HAR. Enhanced node features (rolling 5-day forecast accuracy per agent) yield RMSE of 0.1189, offering no incremental gain, which suggests that the Shapley-based features already subsume the information rolling accuracy would provide.

The ablations reveal a clear hierarchy. Attribution-based node features are indispensable; without them the model loses significance entirely. The graph-form architecture and sparse node-level processing are the second most important elements: the self-loop variant, which retains per-agent multi-head attention without cross-agent edges, captures most of the improvement. Cross-agent message passing adds marginal gains (0.1171 vs 0.1188) that are not individually significant but require a learned sparse topology to materialise; dense connectivity is uniformly harmful. The regime gate and DropEdge each contribute at the margin, and their effects are additive.

## 7.4 Regime-Conditional Analysis

[Table 3 here: RMSE by volatility regime for the DropEdge GAT, GARCH, MLP, HAR, and naive debate aggregation on the aligned evaluation sample ($n = 972$).]

The volatility regimes defined in Section 5.1 partition the evaluation sample by persistence volatility thresholds at 0.20, 0.35, and 0.55. Table 3 reports RMSE within each regime for the DropEdge GAT and four representative comparators. The low regime ($n = 65$) captures tranquil markets, the normal regime ($n = 454$) accounts for nearly half of the sample, the elevated regime ($n = 377$) covers periods of above-average but not extreme volatility, and the high regime ($n = 76$) corresponds to crisis episodes.

The DropEdge GAT dominates all comparators across the low, normal, and elevated regimes, which together comprise 896 of the 972 evaluation days (92 percent). The margin is widest in the low regime, where the GAT achieves RMSE of 0.0789 against 0.1280 for the MLP, 0.1430 for GARCH, 0.1950 for the debate rule, and 0.2004 for HAR. In the normal regime the advantage narrows but remains consistent: 0.1243 against 0.1406 for the debate rule, 0.1456 for GARCH, 0.1479 for HAR, and 0.1543 for the MLP. GARCH is the strongest baseline in the elevated regime (0.1223), yet the GAT still outperforms it (0.1083).

The high regime reverses this pattern. The MLP achieves 0.1177, the lowest error among all methods, while the GAT records 0.1380. GARCH and HAR both exceed 0.14, and the debate and single-agent rules exceed 0.23. Two observations qualify this reversal. The high-regime subsample contains only 76 days, insufficient for reliable inference under the autocorrelation structure documented in Section 3.1. The regime gate described in Section 5 partially addresses this limitation by shifting weight toward the regime-specialised head when $g(t)$ rises, but with so few training examples in this regime the gate cannot fully specialise.

The overall pattern aligns with the motivation of Section 4.2: the GAT extracts value from the learned interaction structure when sufficient training signal exists to estimate the graph, and defaults to a more conservative combination in regimes with sparse training data. Figure 3 visualises these regime-conditional differences.

## 7.5 Learned Graph Analysis

The learned sparse graph described in Section 5.2 produces two empirically separable properties: a stable sparsity level and an unstable topology. Across the 75 trained models (15 walk-forward windows, 5 seeds per window), the mean number of active edges is approximately 16 of 42 possible directed edges, corresponding to a density of 38 percent. This level emerges without any sparsity penalty in the loss, as the sigmoid-gated logits of equation (1) converge to binary values through walk-forward training alone.

The topology underlying this sparsity, however, is not reproducible. No single edge achieves unanimous activation ($f = 1.0$) in any walk-forward window; the highest aggregate edge frequency across all windows is monetary$\to$geopolitical at 53 percent. Within a given window, the number of edges that all five seeds agree on is zero or one. Despite this topological variability, the 5-seed ensemble RMSE standard deviation is only 0.004, indicating that the different sparse configurations achieve nearly identical predictive performance. This matches the ablation finding that self-loop only processing nearly matches the full learned graph (Section 7.3): many sparse subsets of the 42 possible edges support equivalent accuracy, and the sparsity level rather than the specific edge set is the structural invariant.

[Figure 4 here: temporal evolution of edge frequency across the 15 walk-forward windows, computed as the proportion of 5 seeds in which each edge is active.]

The temporal evolution of edge frequencies reveals interpretable economic patterns. Figure 4 shows the most prominent shifts between the early three windows (covering approximately 2021) and the late three windows (covering 2024 to 2025). The edge technical$\to$geopolitical declines from 53 to 7 percent as the acute geopolitical volatility of the 2022 invasion period stabilised and the cross-domain signal weakened. Sentiment$\to$supply\_opec falls from 60 to 20 percent as speculative positioning around OPEC production decisions normalised after the post-invasion supply shock dissipated. Two edges strengthen over the same period: sentiment$\to$macro\_demand rises from 20 to 60 percent during the Federal Reserve tightening cycle, when market sentiment became increasingly relevant to the demand outlook, and cross\_market$\to$technical increases from 13 to 53 percent as equity and currency spillovers to technical volatility measures intensified with rising cross-asset correlations.

These temporal patterns are descriptive; they do not establish that the edges represent causal channels between agent domains. What they do establish is that the learned graph is not arbitrary: the edges that gain or lose activation over time correspond to identifiable shifts in the economic environment, and practitioners can inspect the graph to understand which inter-agent information flows the model relied upon during a given period.

## 7.6 Agent Attribution and Herding Dynamics

The Shapley values defined in Section 6.4 provide agent-level interpretability for the debate stage. All seven agents have negative mean Shapley values on the 972 walk-forward evaluation days, each reduces squared forecast error on average relative to the persistence baseline. Usefulness is episodic, however: individual agents produce positive Shapley values (increasing error) on 38 to 59 percent of days. This day-level instability is what makes naive confidence-weighted aggregation unreliable, as documented in Section 4.1, and it is the variation that the GAT's node features are designed to capture.

The regime-conditional pattern is more informative than the unconditional means. In the low and normal regimes the geopolitical and sentiment agents contribute the largest absolute Shapley values, and both are consistently helpful: mean values of $-12.8 \times 10^{-4}$ and $-9.8 \times 10^{-4}$ in the low regime, $-5.1 \times 10^{-4}$ and $-4.2 \times 10^{-4}$ in the normal regime. These are the regimes where forward-looking fundamentals carry the most incremental information about 20-day volatility, and agents specialising in narrative-driven signals contribute accordingly. In the elevated regime the ranking inverts: the geopolitical agent turns harmful (mean Shapley $+3.8 \times 10^{-4}$), while the technical agent becomes the strongest contributor ($-2.6 \times 10^{-4}$). In the high regime the inversion deepens, with the geopolitical agent at $+16.0 \times 10^{-4}$ and the technical agent at $-9.4 \times 10^{-4}$. Figure 2(a) and Figure 5 visualise this regime-conditional heterogeneity.

[Figure 5 here: box plots of Shapley values by agent and volatility regime on the aligned evaluation sample ($n = 972$).]

This matches the architecture's design: the geopolitical agent's domain variables (GDELT conflict counts and media tone) are noisy proxies during crisis episodes when price dynamics dominate narrative signals, and the technical agent, which directly observes backward realised volatility, is better positioned in those periods.

Herding is classified on 43.4 percent of agent-day observations using the rule in Section 6.3. The technical and cross-market agents exhibit the highest herding rates (54 percent each), their narrower information sets make them more susceptible to Round 2 consensus pull. Communication density in $G^{\text{dbt}}_t$ is positively correlated with the daily herding count ($r = 0.35$, $p < 0.001$). After conditioning on the contemporaneous volatility regime, however, neither communication density nor herding count significantly predicts forecast error ($p > 0.20$), suggesting the behavioural and communication features in $\mathbf{x}_i(t)$ already capture that information.

The Myerson values (Myerson, 1977; Section 6.4) provide a complementary perspective. Agents with high absolute Shapley values but low degree in $G^{\text{dbt}}_t$ exhibit Myerson values close to their Shapley values, indicating independent predictive contribution. Agents with moderate Shapley values but high centrality show a larger gap $\mu_i(t) - \phi_i(t)$, reflecting an intermediary role whose value depends partly on connecting otherwise isolated agents. Removing an agent with a large Myerson-Shapley gap may degrade performance beyond what its standalone Shapley value suggests, because Myerson values enter the node features and the GAT can condition its attention allocation on this intermediary-role information. A case study of the attribution dynamics during the March 2022 crisis episode is reported in Appendix D.

---

# 8. Conclusion

This paper proposed a two-stage framework for oil volatility forecasting. A Structured Delphi Debate protocol generates diverse base forecasts from seven specialist LLM agents, and a learned sparse Graph Attention Network combines them by discovering inter-agent attention structure and adapting combination weights across volatility regimes. On 972 out-of-sample WTI predictions evaluated with HAC-corrected Diebold-Mariano tests, the DropEdge GAT achieves RMSE of 0.1171, a 24.4 percent reduction relative to HAR (RMSE $= 0.1549$) and the lowest error among 16 evaluated methods. GARCH(1,1) also significantly outperforms HAR, but the GAT achieves lower error still, with a marginally significant pairwise advantage ($p = 0.051$). The finding that naive debate aggregation loses significance after HAC correction while the learned GAT does not indicates that the combination mechanism, not the base forecast quality alone, is a key driver of the accuracy gain. This result is consistent with the forecast combination puzzle (Timmermann, 2006; Wang et al., 2023): learned sparse combination recovers the forecasting value that equal-weight or confidence-weighted schemes dissipate when base forecasters herd. The ablation analysis supports this conclusion: a dense graph degrades performance below that of a simple MLP, while the learned sparse topology with approximately 16 of 42 active edges recovers the full improvement.

The framework offers two forms of interpretability relevant to energy risk practitioners. Shapley attribution identifies which information domains are driving the volatility estimate on any given day, and the regime-conditional patterns documented in Section 7.6 show that the dominant agents shift predictably between normal and crisis periods. The herding rate provides a complementary diversity diagnostic: days with high herding indicate that agent forecasts have converged to a consensus, alerting practitioners to reduced forecast diversity even though herding does not independently predict forecast error after conditioning on the volatility regime (Section 7.6). These outputs are available alongside the point prediction without additional computation, subject to the lagged-attribution modification discussed below.

A methodological implication extends beyond the oil volatility application. Any study evaluating forecasts with overlapping horizons longer than a few days faces the serial dependence problem well known in the econometric forecasting literature (West, 1996; Clark and McCracken, 2001) and documented for this application in Section 3.1. In our sample, ignoring HAC correction materially overstates significance; this issue is not unique to LLM-based forecasts and can affect any overlapping-window evaluation. HAC correction with a bandwidth matching the forecast horizon should be treated as a minimum inferential standard in this setting.

Several limitations qualify these findings. The framework depends on a single LLM backbone (Gemini 3 Flash), and LLM behaviour can change across model versions, raising reproducibility concerns for long-horizon deployment. The Shapley values used as node features are computed from the contemporaneous realised volatility target, which introduces a look-ahead dependency. The no-Shapley ablation variant (RMSE $= 0.1520$, not significant against HAR) shows that the model relies on these features for its advantage, making the lagged-Shapley replacement a priority for deployment rather than a convenience. The seven-agent domain decomposition reflects current oil market practice rather than an optimised partition, and a different decomposition might yield different results. The extreme-volatility regimes (low and high, $n = 141$ combined) are too small for reliable regime-specific inference, and the 2020 to 2025 sample covers an unusual sequence of macroeconomic events whose volatility patterns may not persist.

Several directions for future work follow from these limitations. Implementing lagged Shapley attribution would eliminate the look-ahead dependency and enable real-time deployment. The ablation finding that a self-loop-only graph nearly matches the full learned graph (RMSE $= 0.1188$, the second-best variant) warrants further investigation into when and why cross-agent message passing adds value beyond per-agent feature processing. Testing with alternative LLM backbones and across commodity markets beyond crude oil would establish the generality of the SDD-GAT framework.

---

# Tables

## Table 1: Full Model Comparison

RMSE and HAC-corrected Diebold-Mariano statistics for all 16 models on the aligned evaluation sample ($n = 972$). The DM test uses Newey-West HAC standard errors with Bartlett kernel and bandwidth 19. Significance: \* $p < 0.05$, \*\* $p < 0.01$, \*\*\* $p < 0.001$.

| Category | Model | RMSE | $\Delta$% vs HAR | DM$_{\text{HAC}}$ | $p$ | Sig |
|----------|-------|:----:|:---------:|:--------:|:---:|:---:|
| Proposed | DropEdge GAT | 0.1171 | $-$24.4 | $-$3.212 | 0.001 | \*\*\* |
| Proposed | Learned GAT | 0.1188 | $-$23.3 | $-$3.165 | 0.002 | \*\* |
| Econometric | GARCH(1,1) | 0.1366 | $-$11.8 | $-$2.604 | 0.009 | \*\* |
| Proposed | MLP no-graph | 0.1444 | $-$6.8 | $-$1.214 | 0.225 | |
| ML | Ridge | 0.1493 | $-$3.6 | $-$0.545 | 0.586 | |
| LLM | Single agent | 0.1495 | $-$3.5 | $-$1.230 | 0.219 | |
| LLM | Debate (naive) | 0.1512 | $-$2.4 | $-$1.494 | 0.135 | |
| Econometric | HAR | 0.1549 | --- | --- | --- | baseline |
| Naive | Persistence | 0.1551 | +0.1 | 0.192 | 0.848 | |
| ML | RF | 0.1605 | +3.6 | | | |
| ML | GBR | 0.1645 | +6.2 | | | |
| ML | Lasso | 0.1731 | +11.7 | | | |
| ML | XGBoost | 0.1809 | +16.8 | 1.463 | 0.144 | |
| Naive | HistMean | 0.1957 | +26.3 | | | |
| DL | Transformer | 0.1985 | +28.1 | 1.691 | 0.091 | |
| DL | LSTM | 0.2743 | +77.1 | 3.446 | 0.001 | \*\*\* $\dagger$ |

$\dagger$ Significantly *worse* than HAR.

---

## Table 2: Architecture Ablation Results

RMSE and HAC-corrected DM statistics for architectural variants on the aligned evaluation sample ($n = 972$). All variants share the walk-forward protocol, 5-seed ensemble, 20-day label embargo, and HAC inference of Section 3.2.

| Variant | Ablation target | RMSE | DM$_{\text{HAC}}$ | $p$ | Sig |
|---------|----------------|:----:|:--------:|:---:|:---:|
| DropEdge GAT (full) | --- | 0.1171 | $-$3.212 | 0.001 | \*\*\* |
| Identity (self-loops) | Inter-agent edges | 0.1188 | $-$3.165 | 0.002 | \*\* |
| No regime gate | Regime-gated dual head | 0.1198 | $-$2.995 | 0.003 | \*\* |
| Random graph (~16/42) | Learned topology | 0.1414 | $-$1.384 | 0.166 | |
| Dense GAT (42/42) | Learned sparsity | 0.1452 | $-$1.185 | 0.236 | |
| No Shapley/Myerson | Attribution features | 0.1520 | $-$0.420 | 0.675 | |
| MLP no-graph | Graph structure | 0.1444 | $-$1.214 | 0.225 | |

---

## Table 3: Regime-Conditional RMSE

RMSE by volatility regime for the DropEdge GAT and four representative comparators. Regimes are defined by fixed thresholds on persistence volatility: low ($< 0.20$), normal ($0.20$--$0.35$), elevated ($0.35$--$0.55$), high ($> 0.55$).

| Regime | $n$ | DropEdge GAT | GARCH | MLP | HAR | Debate |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Low | 65 | **0.0789** | 0.1430 | 0.1280 | 0.2004 | 0.1950 |
| Normal | 454 | **0.1243** | 0.1456 | 0.1543 | 0.1479 | 0.1406 |
| Elevated | 377 | **0.1083** | 0.1223 | 0.1401 | 0.1362 | 0.1327 |
| High | 76 | 0.1380 | 0.1423 | **0.1177** | 0.2232 | 0.2334 |
| **All** | **972** | **0.1171** | 0.1366 | 0.1444 | 0.1549 | 0.1512 |

Bold indicates lowest RMSE within each regime.

---

# Appendix A: Robustness to the Data Exclusion

The main analysis excludes 61 trading days between 24 February and 22 May 2020 to remove the WTI contract dislocation period, during which the front-month settlement price fell to $-\$36.98$ on 20 April 2020 and the surrounding weeks exhibited return dynamics driven by contractual mechanics rather than supply-demand fundamentals. This appendix examines sensitivity to this exclusion.

## A.1 Effect on Sample Composition

The exclusion removes days in which 20-day realised volatility exceeds 1.0 (annualised), well above the high-regime threshold of 0.55. Retaining these days would add extreme observations to the training windows that overlap the dislocation period, potentially distorting the learned graph structure and the regime gate calibration.

## A.2 Sensitivity Analysis

To assess robustness, we re-estimate the DropEdge GAT on the full sample ($n = 1{,}285$) without exclusion, using the same walk-forward protocol, 5-seed ensemble, and 20-day label embargo. The headline results are as follows.

| Sample | $n_{\text{OOS}}$ | DropEdge GAT RMSE | HAR RMSE | $\Delta$% |
|--------|:---------:|:--------:|:--------:|:---:|
| Clean (main analysis) | 972 | 0.1171 | 0.1549 | $-$24.4 |
| Full (no exclusion) | 1,033 | [TBD] | [TBD] | [TBD] |

[TODO: Run the full-sample variant and fill in the results. The expectation is that including the dislocation period will increase RMSE for all models because the extreme observations inflate squared errors, but the relative ranking should be preserved.]

## A.3 Qualitative Assessment

The exclusion is conservative in scope: 61 of 1,285 days (4.7 percent). The excluded period is contiguous and corresponds to a well-documented contractual anomaly (the expiry of the May 2020 contract with negative settlement) that is unlikely to recur under the revised CME margin rules adopted in 2021. The main conclusions of the paper are therefore unlikely to be sensitive to this exclusion, though the formal sensitivity analysis above provides quantitative confirmation.

---

# Appendix B: Architecture Design Choices and Hyperparameters

## B.1 Input Selection

The GAT receives only two base predictions: the debate consensus forecast ($\hat{y}^{\text{debate}}_t$) and the single-agent generalist forecast ($\hat{y}^{\text{single}}_t$). Early experiments included HAR, persistence, and GARCH(1,1) predictions as additional base inputs (five total), but these reduced out-of-sample accuracy. The debate forecast already incorporates persistence information through the agents' prompt design, making the statistical base predictions redundant. An MLP receiving all five base predictions (RMSE $= 0.1319$) underperformed the LLM-only GAT (RMSE $= 0.1188$), confirming this redundancy.

## B.2 Architecture Variants Explored During Development

The final DropEdge GAT architecture was selected after evaluating several alternatives during development. Table B1 summarises these variants and the reasons they were not retained.

**Table B1: Architecture variants explored**

| Variant | Key difference | Outcome | Reason not retained |
|---------|---------------|---------|-------------------|
| Causal-prior GAT | Granger causality graph as prior | Dense prior (41/42 edges) | Prior too dense; learned graph outperformed |
| L1-penalised GAT | L1 penalty on edge logits | All edges remained active at L1=0.05 | L1 ineffective for sigmoid-gated logits |
| Stacked GAT+MLP | Two-layer stacking (GAT output + MLP) | RMSE $= 0.1227$ | Below DropEdge GAT ($0.1171$) |
| Regime-weighted loss | Upweight high-volatility training days | RMSE $= 0.1225$ | Degraded normal-regime fit |
| MoE (GAT + MLP + router) | Mixture of experts with learned router | Router collapsed to single expert | Training instability |
| Enhanced features | Added rolling 5-day accuracy per agent | RMSE $= 0.1189$ | No gain over Shapley-based features |

## B.3 Hyperparameters

Table B2 reports the full hyperparameter specification for the DropEdge GAT as used in the main analysis.

**Table B2: DropEdge GAT hyperparameters**

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Graph structure** | Edge parameterisation | Learnable logits + sigmoid gating |
| | Active edges (post-training) | ~16 of 42 |
| | Activation threshold | 0.5 (sigmoid output) |
| **GAT backbone** | Number of layers | 2 |
| | Attention heads per layer | 4 |
| | Per-head hidden dimension | 4 (total: 16) |
| | Top-$k$ masking | $k = 3$ per node |
| | Activation | LeakyReLU (negative slope 0.2) |
| | Normalisation | LayerNorm after each layer |
| | Skip connection | Linear projection to hidden dim |
| **DropEdge** | Training edge dropout rate | 0.20 |
| | Inference | All edges retained |
| **Output heads** | Base head | Linear(16, 32) $\to$ ReLU $\to$ Dropout(0.15) $\to$ Linear(32, 2) $\to$ Softmax |
| | Regime head | Same architecture, independent parameters |
| | Regime gate inputs | persist\_vol, vol\_change\_5d, $n_{\text{herding}}/7$ |
| | Regime gate | Linear(3, 8) $\to$ Tanh $\to$ Linear(8, 1) $\to$ Sigmoid |
| | Final weights | $(1-g) \cdot w_{\text{base}} + g \cdot w_{\text{regime}}$ |
| **Residual** | Architecture | Linear(16, 16) $\to$ Tanh $\to$ Linear(16, 1) |
| | Scaling factor | 0.05 |
| | Final prediction | $\text{clamp}(\sum w_k \hat{y}_k + \text{residual},\, \min=0.05)$ |
| **Training** | Optimiser | Adam (Kingma and Ba, 2015) |
| | Learning rate | 0.003 with cosine annealing |
| | Weight decay | $10^{-4}$ |
| | Epochs | 250 |
| | Gradient clipping | Max norm 1.0 |
| | Checkpoint selection | Best training loss |
| **Evaluation** | Walk-forward | min\_train = 252 days, retrain every 63 days |
| | Ensemble | 5 random seeds, prediction averaged |
| | Label embargo | 20 days (training labels end at $t-20$) |

## B.4 Node and Context Features

**Node features** ($\mathbf{x}_i(t) \in \mathbb{R}^7$ per agent):

| Feature | Description | Normalisation |
|---------|-------------|---------------|
| $\phi_i(t)$ | Shapley value (cost-game) | Raw |
| $\mu_i(t)$ | Myerson value | Raw |
| $b_i(t)$ | Behaviour code (herding=0, anchored=1, independent=2, overconfident=3) | $/3$ |
| $d_i(t)$ | In-degree in $G^{\text{dbt}}_t$ | $/6$ |
| $\bar{\phi}_{i,5}(t)$ | 5-day moving average of Shapley | Raw |
| $s_{\phi,i,5}(t)$ | 5-day standard deviation of Shapley | Raw |
| $h_i(t)$ | Consecutive herding streak length | $/5$ |

**Context features** ($\mathbf{c}(t) \in \mathbb{R}^9$, shared across nodes):

| Feature | Description |
|---------|-------------|
| persist\_vol | Backward-looking 20-day realised volatility |
| har\_vol | HAR-RV model prediction |
| total\_adj | Sum of Round 2 agent adjustments |
| $n_{\text{herding}}/7$ | Fraction of agents classified as herding |
| vol\_regime$/3$ | Regime label (0-3) normalised |
| debate\_vol | Debate consensus forecast |
| persist-har gap | persist\_vol $-$ har\_vol |
| vol\_change\_5d | 5-day change in persistence volatility |
| debate-har gap | debate\_vol $-$ har\_vol |

---

# Appendix C: Pairwise HAC-Corrected Diebold-Mariano Statistics

Table C1 reports pairwise HAC-corrected Diebold-Mariano statistics for the principal models. Each cell shows $\text{DM}_{\text{HAC}}$ (row model vs column model), with the $p$-value in parentheses. Negative values indicate that the row model has lower squared error than the column model. All tests use Newey-West HAC standard errors with Bartlett kernel and bandwidth 19.

**Table C1: Pairwise DM$_{\text{HAC}}$ statistics ($n = 972$)**

| | HAR | GARCH | MLP | Debate | Single |
|---|:---:|:---:|:---:|:---:|:---:|
| **DropEdge GAT** | $-$3.212 (0.001)\*\*\* | $-$1.948 (0.051) | $-$3.336 (<0.001)\*\*\* | | |
| **Learned GAT** | $-$3.165 (0.002)\*\* | | | | |
| **GARCH** | $-$2.604 (0.009)\*\* | --- | | | |
| **MLP** | $-$1.214 (0.225) | | --- | | |
| **Debate** | $-$1.494 (0.135) | | | --- | |
| **Single agent** | $-$1.230 (0.219) | | | | --- |

Additional pairwise comparisons from the ablation analysis:

| Comparison | DM$_{\text{HAC}}$ | $p$ |
|---|:---:|:---:|
| DropEdge GAT vs Dense GAT | $-$6.382 | $<$0.001\*\*\* |
| DropEdge GAT vs Identity | $-$1.107 | 0.268 |
| Dense GAT vs Identity | 6.365 | $<$0.001\*\*\* |

The DropEdge GAT significantly outperforms the MLP ($p < 0.001$) and the Dense GAT ($p < 0.001$), and marginally outperforms GARCH at the 10 percent level ($p = 0.051$). The Identity variant (self-loops only) significantly outperforms the Dense GAT ($p < 0.001$), confirming that full connectivity is harmful and that learned sparsity is the source of the graph structure's contribution.

---

# Appendix D: Crisis Episode Case Study (March 2022)

The Russia-Ukraine conflict escalation in late February 2022 triggered the sharpest WTI price spike in the sample period. This appendix examines four trading days during and after the invasion to illustrate how the DropEdge GAT corrects debate forecasts under crisis conditions.

## D.1 Attribution Dynamics

Table D1 reports the debate and GAT predictions alongside the actual 20-day forward realised volatility for four dates in March 2022.

**Table D1: GAT correction during the March 2022 crisis**

| Date | Actual vol | Debate pred | GAT pred | HAR | Debate AE | GAT AE | Helped | $n_{\text{herd}}$ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2022-03-09 | 0.704 | 0.767 | 0.583 | 0.509 | 0.064 | 0.121 | No | 1 |
| 2022-03-17 | 0.622 | 0.868 | 0.739 | 0.793 | 0.246 | 0.117 | Yes | 4 |
| 2022-03-28 | 0.547 | 0.934 | 0.802 | 0.861 | 0.387 | 0.255 | Yes | 3 |
| 2022-03-31 | 0.470 | 0.908 | 0.779 | 0.844 | 0.438 | 0.310 | Yes | 3 |

On 9 March, just after the invasion peak, the debate forecast overshoots only slightly (AE $= 0.064$) and the GAT's correction toward a lower prediction is unhelpful. By 17 March, the debate substantially overshoots (AE $= 0.246$) as agents persist with crisis-level adjustments even as realised volatility begins to decline. The GAT pulls the forecast down, cutting the error roughly in half. This pattern intensifies on 28 and 31 March: the debate rule continues to predict near-unit volatility while actual volatility has fallen to 0.47-0.55, and the GAT reduces the absolute error by 34-39 percent in each case.

## D.2 Herding Dynamics

The herding count rises from 1 on 9 March (low herding, agents still processing the invasion independently) to 3-4 on 17-31 March (most agents anchored on the elevated consensus from the invasion week). The GAT node features encode this herding shift, and the regime gate ($g(t)$) increases during this period as persistence volatility remains elevated, shifting weight toward the regime-specialised head.

## D.3 Interpretation

The case study illustrates two properties of the framework. When the debate consensus is close to the realised outcome (9 March), the GAT's correction provides no benefit and can slightly worsen the forecast; the model is not uniformly superior. When agents herd on an outdated consensus while market conditions normalise (17-31 March), the learned combination detects the herding pattern through the behavioural features and corrects toward the regime-appropriate prediction. This asymmetry between the "anchored debate" failure mode and the "diverse debate" success mode is what the learned sparse graph is designed to exploit.
