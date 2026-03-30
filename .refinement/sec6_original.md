# Section 6 Original (from outline_ijf.md lines 513-586)
# Saved as backup before refinement

## 6. Structured Delphi Debate Protocol (~1.5-2 pages)

### 6.1 Agent design

Seven specialist agents are instantiated from Gemini 3 Flash, each
provided with a distinct structured feature set aligned with a
recognised oil market analysis domain.
The agents cover: geopolitical factors (GDELT event features), macro
demand (VIX, equity indices), monetary conditions (federal funds
rate, yield curve slope), supply and OPEC (production and inventory
signals), technical price behaviour (momentum and moving-average
features), sentiment (market implied volatility structure), and
cross-market linkages (DXY, credit spreads).
Each agent outputs four structured fields: a point adjustment to the
persistence forecast (in the same units as the target), a confidence
score in [0, 1], a directional label, and a structured evidence
summary.
The structured output format is enforced via JSON schema constraints.
This design adapts the Delphi elicitation method to a reproducible
multi-agent LLM inference procedure (Dalkey and Helmer, 1963).

### 6.2 Two-round debate protocol

In Round 1, each agent receives only its own feature set and
produces an independent forecast adjustment.
This guarantees the diversity of the initial forecasts, preventing
premature consensus.
In Round 2, each agent receives all seven Round 1 adjustments and
their accompanying evidence summaries, and is asked to revise its
own adjustment given this additional context.
The final base forecasts delivered to Stage 2 are
$\hat{y}^{\text{debate}}_t$ (confidence-weighted Round 2 mean) and
$\hat{y}^{\text{single}}_t$ (Round 1 single-agent average without
debate).

### 6.3 Behavioural classification

Each agent at each time step is classified into one of four
behavioural states based on its Round 1 to Round 2 adjustment
trajectory.
Herding occurs when the adjustment moves toward the Round 1 group
mean.
Anchoring occurs when the absolute adjustment changes by less than
a minimum threshold.
Independent behaviour occurs when the adjustment changes
substantially in a direction not explained by the group mean.
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
independently valuable from those whose value depends on their
network position.
The results from applying this framework are reported in Section 7.
