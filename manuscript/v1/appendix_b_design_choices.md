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
