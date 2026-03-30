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
Restricting Stage 2 inputs to these two LLM-derived quantities is
deliberate: experimentation confirmed that adding statistical
model outputs (HAR, GARCH, persistence) as additional base forecasts
reduces accuracy, because $\hat{y}^{\text{debate}}_t$ already
incorporates persistence through the agents' prompt design, making
the statistical signals redundant rather than informative.

Each node $a_i$ carries a 7-dimensional feature vector
$\mathbf{x}_i(t) \in \mathbb{R}^7$,
assembled from time-$t$ quantities that characterise the agent's
recent behaviour and influence.
The seven components are:
the Shapley value $\phi_i(t)$ (cost-game convention, so negative
values indicate that the agent reduces forecast error);
the Myerson value $\mu_i(t)$, which accounts for the agent's
position in the influence graph;
a normalized four-category behaviour code $b_i(t)/3$,
where the four categories are herding, anchored, independent,
and overconfident, giving values in $\{0, 1/3, 2/3, 1\}$;
the normalized in-degree $d_i(t)/6$ from the debate influence graph;
the 5-day moving average of $\phi_i$;
the 5-day standard deviation of $\phi_i$; and
the normalized herding streak $s_i(t)/5$,
computed as a rolling 5-day sum of the herding indicator.
The moving average and standard deviation capture trend and
uncertainty in the agent's recent contribution, while the herding
streak distinguishes persistent herding from isolated episodes.

A 9-dimensional context vector $\mathbf{c}(t) \in \mathbb{R}^9$
carries market-state information that is not agent-specific.
Its components are: the persistence volatility $\text{rv}_{20,t}$;
the HAR model volatility estimate;
the total agent adjustment $\hat{y}^{\text{debate}}_t - \text{rv}_{20,t}$;
the normalized herding count $n_{\text{herd}}(t)/7$;
a normalized four-level volatility regime indicator $r(t)/3$,
where the four levels correspond to persistence thresholds at
0.20, 0.35, and 0.55;
the debate forecast $\hat{y}^{\text{debate}}_t$;
the persistence-HAR gap;
the one-day change in persistence volatility; and
the debate-HAR gap.
This vector is concatenated with the graph embedding before the
output head and also supplies the three signals consumed by the
regime gate in Section 5.5.

## 5.2 Graph construction

The interaction topology is not pre-specified.
A learnable logit matrix $\mathbf{L} \in \mathbb{R}^{7 \times 7}$
parameterises the edge set, with $\mathbf{L}_{ij}$ representing the
log-odds of the edge from $a_i$ to $a_j$ being active.
The effective edge weight entering the GAT layers is

$$A_{ij} = \sigma(L_{ij}),\tag{1}$$

where $\sigma$ denotes the logistic sigmoid.
Self-loops are fixed at one throughout training and inference
so that each node always attends to itself.
The logits are initialised by drawing from $\mathcal{N}(0, 0.1^2)$,
placing the initial graph close to uniform probability (0.5) for
every possible edge, so no structural prior is imposed.
After each training run, edges with $A_{ij} < 0.5$ are treated as
inactive; the network consistently converges to roughly 16 of the
42 possible directed edges across seeds and training windows.

This mechanism constitutes the first of three sparsity layers in
the architecture.
The sigmoid gating produces a soft global topology,
and the hard 0.5 threshold applied post-training identifies a sparse
effective graph without any penalty term in the loss.
Appendix B reports experiments with Granger-based causal priors
as warm-start initialisations for $\mathbf{L}$;
those variants do not outperform learning from scratch,
suggesting that the data-driven topology is not simply recovering
a known causal structure.

## 5.3 Multi-head GAT layers

The graph encoder consists of two successive Graph Attention Network
layers (Veličković et al., 2018),
each using four attention heads with a per-head hidden dimension of
four, giving a total hidden dimension of 16.
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
The decomposed additive formulation avoids the full concatenation of
the original GAT paper (Veličković et al., 2018) at no loss in
expressive power for fixed-size node sets.
The edge weight matrix $\mathbf{A}$ from equation (1) enters as a
multiplicative mask: positions with $A_{ij} = 0$ receive a score
of $-10^9$ before softmax, effectively zeroing out the corresponding
attention weight.

The second sparsity layer operates within each attention head:
top-$k$ masking with $k=3$ retains only the three highest-scoring
incoming edges per node, setting the remainder to $-10^9$.
The resulting attention coefficient is

$$\alpha_{ij}^{(k)} = \text{softmax}_{i}\!\left(e_{ij}^{(k)}\right),\tag{3}$$

where the softmax is taken over the (at most $k=3$) unmasked source
nodes.
Node $a_j$'s updated representation is the concatenation of the
head-specific aggregations,
$\mathbf{h}_j' = \bigl\|_{k=1}^{4}
\sum_{i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_i$.

Each GAT layer includes a skip connection:
node features are projected to the hidden dimension by a shared
linear map, added to the attention output, and passed through
LayerNorm before ReLU activation.
The second layer adds its output to the first layer's output rather
than to the raw features, giving the network a residual path
of depth two.

After both layers, the seven node representations are concatenated
rather than pooled,
yielding a graph embedding of dimension $7 \times 16 = 112$.
Concatenation is appropriate because the seven nodes are semantically
distinct: each represents a specialist agent with a fixed and
interpretable domain, so averaging over them would discard the
agent-specific variation that the node features are designed to
capture.
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
The motivation is specific to this setting: Section 7 shows that
the learned topology is not unique.
Multiple sparse graphs of similar cardinality achieve comparable
training loss, with no single edge appearing consistently across
all seeds and training windows.
DropEdge implicitly trains the network on an ensemble of subgraphs,
which improves out-of-sample robustness when the true optimal graph
is not identifiable from the available data.

DropEdge also resolves a design tension encountered in earlier
model variants.
$L_1$ regularisation on the edge logits was tested as a way to
encourage sparsity but failed: even with a penalty coefficient of
0.05, the network retained 41 of 42 possible edges because the
penalty acts on logit magnitudes rather than on the binary edge
existence decisions.
The learned-sigmoid mechanism in Section 5.2 achieves sparsity
through the saturating nonlinearity alone, and DropEdge then
regularises without interfering with that mechanism.
These earlier variants and their results are documented in
Appendix B.

## 5.5 Regime-gated dual-head output

The output layer converts the combined representation $\mathbf{z}(t)$
into a convex combination of the two base forecasts.
Two independent heads, referred to as the base head and the regime
head, each map $\mathbf{z}(t)$ through a 32-unit hidden layer with
ReLU activation and 15% dropout, then through a linear layer to two
logits, and finally through softmax to produce a weight vector over
$[\hat{y}^{\text{debate}}_t,\, \hat{y}^{\text{single}}_t]$.
The heads share the same architecture but have entirely separate
parameters.

A regime gate determines which head governs the final combination.
The gate takes three scalars as input: the persistence volatility
$\text{rv}_{20,t}$, the one-day change in persistence volatility
$\Delta\text{rv}_{20,t}$, and the normalized herding count
$n_{\text{herd}}(t)/7$.
These three signals pass through a two-layer network (8 hidden units,
Tanh activation) followed by a sigmoid output, giving a scalar gate
$g(t) \in [0,1]$.
The final combination weights are

$$\mathbf{w}(t) = (1 - g(t))\, \mathbf{w}_{\text{base}}(t)
+ g(t)\, \mathbf{w}_{\text{regime}}(t).\tag{4}$$

When $g(t)$ is near zero the base head dominates,
producing a combination that is primarily driven by the
agent-interaction patterns encoded in the graph.
When $g(t)$ is near one the regime head takes over,
which is intended to capture periods where the market regime is
itself the primary determinant of which forecast to trust.

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
volatility of 5%, consistent with the range observed in the sample.

## 5.6 Training protocol

The DropEdge GAT is trained with Adam (Kingma and Ba, 2015),
with a learning rate of 0.003 and weight decay of $10^{-4}$.
A cosine annealing schedule reduces the learning rate from 0.003
to zero over 250 epochs.
Gradient norms are clipped to 1.0 at each step to prevent
instability during the early phase of graph learning,
when the edge logits are far from convergence and can produce
large gradient signals.
The checkpoint with minimum training loss across the 250 epochs
is retained; validation-based early stopping was tested but caused
underfitting at the training window sizes encountered in the
walk-forward protocol.

Training is embedded within the walk-forward evaluation protocol
of Section 3.2.
A new model instance is trained from scratch at each retraining
point, every 63 trading days, using all available clean observations
up to that point subject to the 20-day label embargo.
Five independent random seeds initialise both the edge logits and
the network weights; reported metrics are 5-seed ensemble means,
following the practice of (Lakshminarayanan et al., 2017) for
uncertainty reduction through deep ensemble averaging.

Section 6 describes how the debate forecast $\hat{y}^{\text{debate}}_t$
and the single-agent forecast $\hat{y}^{\text{single}}_t$ consumed
by equation (5) are generated.
