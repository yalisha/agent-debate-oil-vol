# 3. Problem Formulation and Notation

## 3.1 Prediction Target

Let $r_t = \log(P_t / P_{t-1})$ denote the daily log return on WTI crude oil,
where $P_t$ is the settlement price on trading day $t$.
The prediction target is the forward-looking 20-day realised volatility,

$$\text{fwd\_rv}_{20,t} = \sqrt{252} \cdot
\text{std}\!\left(r_{t+1}, \ldots, r_{t+20}\right),$$

an annualised volatility expressed as a decimal.
The 20-day horizon is chosen to match the VaR and delta-hedging horizons
used widely in energy risk management,
and it aligns the forecast with the forward period over which a
portfolio manager would wish to hold a volatility hedge.

Because the estimation window $[t+1, t+20]$ advances by a single day
between consecutive forecasts,
successive target values share nineteen of their twenty constituent returns.
This overlap induces strong serial correlation in forecast errors
by construction: the empirical first-order autocorrelation of the error
series is 0.938.
The implication for inference is material.
Standard Diebold-Mariano tests assume short-range dependence,
and applying them without correction to an overlapping-horizon series
overstates statistical significance by approximately 65%, as documented
in Section 7.
All inference in this paper therefore relies on Newey-West
HAC-corrected Diebold-Mariano tests with a Bartlett kernel
and bandwidth 19, following the recommendation of Harvey, Leybourne
and Newbold (1997) for overlapping forecast evaluation.

## 3.2 Evaluation Protocol

Forecast accuracy is assessed using a walk-forward protocol that
mirrors realistic deployment constraints.
The initial training window spans 252 trading days,
and the model is retrained every 63 trading days thereafter,
corresponding to approximately one calendar quarter.
The full sample covers January 2020 to May 2025, yielding 1,285
trading days in total.
Sixty-one days on which realised volatility exceeds 2.0 are excluded
from evaluation; these correspond primarily to the April 2020
negative-price episode, when WTI settlement fell to $-\$36.98$,
and the roughly 20 trading days before and after it during which
return variation was contaminated by the resulting price dislocations.
The clean sample contains 1,224 observations.
After reserving the minimum training window, 972 out-of-sample
predictions are available for evaluation.
All results reported in Section 7 are computed on these 972 predictions.

A further constraint governs the supervised learning components of
the framework.
The target label for day $t$ is not fully observable until day $t+20$,
because the forward window closes only then.
Training a supervised model up to evaluation boundary $t$ using
labels through $t-1$ would therefore allow the model to see
partially-realised future volatility during training,
inflating apparent accuracy relative to what is achievable
in real time.
To prevent this, a 20-day label embargo is imposed throughout:
any supervised model evaluated at boundary $t$ uses training labels
only up to $t - 20$.
The embargo applies uniformly to all supervised baselines and to the
GAT meta-aggregator described in Section 5.
The DropEdge GAT is a five-seed ensemble,
and all reported statistics are ensemble means across seeds.

## 3.3 Notation

The following notation is used throughout the paper.
$\mathcal{A} = \{a_1, \ldots, a_7\}$ denotes the set of seven
specialist LLM agents, each assigned a distinct information domain.
At each evaluation date $t$, every agent $a_i$ produces a Round 1
adjustment $\hat{v}^{(1)}_i(t)$ and, after observing the Round 1
outputs of all other agents, a Round 2 adjustment $\hat{v}^{(2)}_i(t)$.
Both adjustments are expressed relative to the persistence baseline,

$$v^{\text{persist}}_t = \text{fwd\_rv}_{20,\, t-1},$$

the most recent realised volatility observation available at $t$.
The naive aggregation baseline $\hat{y}^{\text{debate}}_t$ is the
confidence-weighted mean of Round 2 adjustments added to persistence:

$$\hat{y}^{\text{debate}}_t =
v^{\text{persist}}_t +
\sum_{i=1}^{7} w_i\, \hat{v}^{(2)}_i(t),$$

where the weights $w_i$ are proportional to the confidence scores
that each agent reports alongside its forecast.
The single-agent baseline $\hat{y}^{\text{single}}_t$ is produced by
one agent operating without access to peer forecasts,
serving as a control for the debate mechanism itself.

The Shapley value $\phi_i(t)$ measures the marginal contribution of
including agent $a_i$'s adjustment in the naive aggregation at time $t$,
computed as the average improvement in forecast accuracy over all
orderings in which $a_i$ could join the active set.
A negative Shapley value indicates that including the agent reduces
squared error, so all seven agents are informative when evaluated
individually.

The influence graph $G_t = (\mathcal{A},\, \mathcal{E}_t,\, \mathbf{W}_t)$
represents the learned combination structure at time $t$,
with edge set $\mathcal{E}_t \subseteq \mathcal{A} \times \mathcal{A}$
and attention weight matrix $\mathbf{W}_t$.
An edge $(a_i, a_j) \in \mathcal{E}_t$ indicates that agent $a_j$'s
node representation attends to agent $a_i$ during graph aggregation.
The edge set is not pre-specified by domain knowledge or a causality
test; instead, learnable sigmoid-gated edge logits allow the network
to discover which connections reduce out-of-sample error,
converging to approximately 16 of 42 possible edges in the trained model.

Section 4 draws on these definitions to show, empirically,
why naive aggregation over $\mathcal{A}$ fails even when every
$\phi_i(t)$ is negative, and how the graph $G_t$ addresses that failure.
