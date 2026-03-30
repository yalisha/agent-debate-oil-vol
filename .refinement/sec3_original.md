# Section 3 Original (Outline Content)

## 3. Problem Formulation and Notation (~1 page)

### 3.1 Prediction target

Let $r_t = \log(P_t / P_{t-1})$ denote the daily WTI log return.
The prediction target is the forward-looking 20-day realised volatility:

$$\text{fwd\_rv}_{20,t} = \sqrt{252} \cdot \text{std}\bigl(r_{t+1}, \ldots, r_{t+20}\bigr)$$

This is an annualised volatility expressed as a decimal.
The 20-day horizon matches practitioner VaR and hedging horizons.
Because the target window overlaps across consecutive trading days, forecast errors are serially correlated by construction, with empirical ACF lag-1 of 0.938.
This motivates the use of Newey-West HAC-corrected DM tests throughout; Section 7 shows that the uncorrected DM statistic overstates significance by approximately 65% relative to the HAC-corrected version.

### 3.2 Walk-forward evaluation

The evaluation follows a strict walk-forward protocol with minimum training window 252 trading days and retraining every 63 trading days.
The full sample spans 2020-01 to 2025-05 (n = 1,285 trading days).
After excluding 61 days with actual_vol > 2 (primarily the April 2020 negative-price episode and its aftermath), the clean sample is n = 1,224.
The minimum training window leaves 972 out-of-sample predictions available for evaluation.
All results reported in Section 7 use these 972 walk-forward predictions.
The GAT meta-aggregator is a 5-seed ensemble; all reported statistics are ensemble means.

Because the target label for day t is not fully observed until day t+20, a 20-day label embargo is imposed: supervised models training up to evaluation boundary t use labels only through t-20.
This prevents label leakage that would otherwise inflate supervised model accuracy.
The same embargo applies to all supervised baselines and to the GAT meta-aggregator.

### 3.3 Notation

$\mathcal{A} = \{a_1, \ldots, a_7\}$ denotes the set of seven specialist agents.
$\hat{v}^{(1)}_i$ and $\hat{v}^{(2)}_i$ denote agent $i$'s Round 1 and Round 2 forecast adjustments relative to the persistence baseline $v^{\text{persist}}_t$.
$\hat{y}^{\text{debate}}_t$ denotes the confidence-weighted mean of Round 2 adjustments added to persistence (the naive aggregation baseline).
$\hat{y}^{\text{single}}_t$ denotes the single-agent baseline (no debate).
$\phi_i(t)$ denotes the Shapley value of agent $i$ at time $t$, defined as the marginal contribution of including agent $i$'s adjustment in the naive aggregation.
$G_t = (\mathcal{A}, \mathcal{E}_t, \mathbf{W}_t)$ denotes the influence graph at time $t$, with edge set $\mathcal{E}_t$ and learned attention weights $\mathbf{W}_t$.
Section 4 uses these definitions to motivate why the naive aggregation baseline fails and why a graph-based approach is needed.
