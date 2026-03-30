# Codex Review for Section 6

Target text reviewed:
- `/Users/mac/computerscience/17Agent可解释预测/.refinement/sec6_round2.md`

Primary reference files:
- `/Users/mac/computerscience/17Agent可解释预测/manuscript/sec3_formulation.md`
- `/Users/mac/computerscience/17Agent可解释预测/manuscript/sec5_architecture.md`
- `/Users/mac/computerscience/17Agent可解释预测/src/debate_system.py`
- `/Users/mac/computerscience/17Agent可解释预测/src/run_debate_eval.py`

## Overall assessment

Section 6 is close to usable. Sections 6.1-6.3 are largely aligned with the v3 implementation and with the current Section 5 narrative. The main remaining problems are concentrated in Section 6.4, where the attribution mathematics in the prose does not fully match the implemented estimator.

## Priority issues

### P1. Coalition forecast in 6.4 does not match the implemented Shapley/Myerson estimator

In the prose, coalition forecasts are described as using only the Round 2 adjustments of agents in `S`, with confidence weights renormalized within the active coalition:

- `sec6_round2.md`, lines 166-171
- `sec3_formulation.md`, lines 98-104

But the implemented attribution engine does something different:

- agents outside the coalition are not removed completely
- instead, they contribute `0` adjustment and receive a minimum weight of `0.01`

Code:
- `src/debate_system.py`, lines 521-532

This changes the meaning and scale of both `phi_i(t)` and `mu_i(t)`.

Recommendation:
- either align the code to the paper's current coalition definition
- or revise Section 6.4 and Section 3.3 so the paper matches the implemented estimator

Do not keep both versions simultaneously.

### P1. Myerson sign interpretation is inconsistent with the cost-game convention

Section 6.4 correctly states that under the cost-game convention, negative values are better:

- `sec6_round2.md`, lines 172-173

But later it says:

- agents whose value depends on network position see `mu_i(t)` "diverge upward" from `phi_i(t)`
- independent agents see `mu_i(t) <= phi_i(t)`

These directional claims are not clean under a cost-game interpretation, because stronger contribution typically means a more negative value, not a larger one.

Recommendation:
- avoid directional inequality claims unless the sign convention is restated carefully
- safer phrasing: compare `mu_i(t)` and `phi_i(t)` in absolute or relative terms without asserting that "upward" necessarily means more contribution

### P2. Myerson connectivity description is slightly stronger than the code

The current prose says Myerson allows only connected coalitions and implies isolated agents receive zero credit.

But the implementation:
- computes connected components in the undirected sense
- then activates only the connected component containing the focal agent

Code:
- `src/debate_system.py`, lines 459-487
- `src/debate_system.py`, lines 565-578

An isolated agent does not receive zero credit in general. It retains the contribution of its singleton component.

Recommendation:
- say "connected in the undirected sense"
- say "isolated agents retain only standalone contribution"

## Suggested fixes for Section 6.4

1. Make the Shapley estimator description match the actual coalition forecast rule.
2. Remove or rewrite the inequality language comparing `mu_i(t)` and `phi_i(t)`.
3. Clarify that the debate influence graph is treated as undirected when computing connected components for Myerson.
4. Clarify that isolation reduces network-mediated credit rather than forcing total credit to zero.

## Secondary notes

- The "500-sample Monte Carlo" statement is consistent with the evaluation pipeline:
  - `src/run_debate_eval.py`, lines 236-237
  - `src/run_debate_eval.py`, line 283
- The two-round protocol is consistent with the evaluation script default:
  - `src/run_debate_eval.py`, line 236
  - `src/run_debate_eval.py`, line 513
- The single-agent baseline description is now consistent with Section 3 and the implementation:
  - `sec6_round2.md`, lines 80-84
  - `src/debate_system.py`, lines 780-805

## Bottom line

Section 6 is nearly converged, but Section 6.4 still needs one more pass before integration into the manuscript. The core issue is not style. It is estimator-definition consistency across:

- `sec3_formulation.md`
- `sec6_round2.md`
- `src/debate_system.py`
