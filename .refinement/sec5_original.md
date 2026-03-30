# Section 5 Original (from outline_ijf.md lines 388-510)

## 5. GAT Architecture (~2 pages)

[Figure 1 here: two-stage architecture diagram]

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
capturing information about the current market environment that
is not agent-specific.
The nine components are: persistence volatility, HAR volatility,
total agent adjustment, normalised herding count, volatility regime
encoded as a three-class one-hot vector, debate forecast,
persistence-HAR gap, 5-day volatility change, and debate-HAR gap.
This vector feeds directly into the regime gate and into the
residual branch described below.

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
(Velickovic et al., 2018), each with four attention heads and
hidden dimension 16.
Within each head, the attention coefficient between nodes $i$ and
$j$ is computed as:

$$\alpha_{ij}^{(k)} = \text{softmax}_j \bigl(
\text{LeakyReLU}(\mathbf{a}^{(k)T}
[\mathbf{W}^{(k)}\mathbf{h}_i \| \mathbf{W}^{(k)}\mathbf{h}_j])
\bigr)$$

where $\mathbf{h}_i$ is node $i$'s hidden state and $\|$ denotes
concatenation.
Top-k masking (k = 3) retains only the three highest-weight
neighbours per node within each head, providing an additional
sparsity constraint at inference.
Each layer applies a skip connection projecting node features to
the hidden dimension, followed by LayerNorm.
The output of the two-layer GAT is a graph-level embedding formed
by mean-pooling over node representations.

### 5.4 DropEdge regularisation

During training, 20% of active edges are randomly dropped at each
forward pass (DropEdge regularisation).
This prevents the network from relying on any single edge
configuration and encourages robustness across the multiple locally
optimal sparse graphs documented in Section 7.5.
At inference, all learned edges are active.
The combination of data-driven edge learning with stochastic edge
dropout resolves a tension present in earlier design variants: L1
sparsity regularisation (Appendix B) failed because it penalises
weight magnitude rather than edge existence, while DropEdge improves
generalisation without distorting the sparsity-finding mechanism.

### 5.5 Regime-gated dual-head output

The combination mechanism uses a dual-head architecture that
separates regime-neutral and regime-sensitive combination.
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
A residual branch (Linear(graph_dim, 16), Tanh, Linear(16, 1),
scaled by 0.05) adds a small graph-informed correction to the
combination.
The final prediction is
$\hat{y}_t = \text{clamp}(\mathbf{w}^T
[\hat{y}^{\text{debate}}_t, \hat{y}^{\text{single}}_t]
+ \text{residual},\; \min = 0.05)$.

### 5.6 Training protocol

The model is trained with Adam (learning rate 0.003,
weight decay 1e-4) using a cosine learning rate schedule over
250 epochs.
Gradient norm clipping at 1.0 prevents instability from
early-stage graph learning.
The checkpoint with minimum training loss is retained.
For the walk-forward evaluation, retraining occurs every 63 trading
days using all available data up to that point (minimum 252 days),
subject to the 20-day label embargo described in Section 3.2.
Five independent random seeds are used; reported metrics are 5-seed
ensemble means.
Section 6 describes how the base forecasts consumed by this
architecture are generated.
