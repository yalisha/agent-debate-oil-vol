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
