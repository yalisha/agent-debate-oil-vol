# 4. Motivation for the GAT Architecture

## 4.1 Why naive aggregation fails

The natural starting point for combining outputs from a multi-agent
debate is the confidence-weighted mean defined in Section 3.3.
After correcting for the serial dependence induced by the 20-day
overlapping forecast horizon, however, this naive rule does not
significantly outperform the HAR benchmark.
The HAC-corrected Diebold-Mariano statistic is $-1.494$ ($p = 0.135$),
meaning the null of equal predictive accuracy cannot be rejected
at conventional levels.
The problem is not simply that the base forecasts are uninformative.
Shapley attribution shows that most agents reduce forecast error on
average, and the debate outputs contain signal that, when
properly extracted, supports substantial accuracy gains.

The obstacle is more specific: not all agents help, and no agent
helps reliably across all market conditions.
The geopolitical agent carries a positive mean Shapley value under
the cost-game convention of Section 3.3, indicating that on average
its inclusion increases squared forecast error rather than reducing it.
The remaining six agents have negative mean Shapley values, but even
they produce positive values on 38 to 56 percent of individual
evaluation days.
Confidence-weighted averaging treats each agent proportionally to
its self-reported certainty, which is not the same as its marginal
predictive contribution.
An agent that is consistently confident but sporadically useful
receives persistent weight rather than selective weight,
and the mismatch between confidence and contribution is exactly what
naive aggregation cannot resolve.

The herding phenomenon documented in Section 6.3 compounds this
problem.
Across the full clean sample, 42.9 percent of agent-day observations
are classified as herding, meaning the agent moved its Round 2
adjustment toward the group median rather than toward a domain-specific
view.
Herding reduces the effective number of independent forecasts below
seven without reducing the nominal weight total.
When the majority of agents converge toward the same position, the
confidence-weighted mean amplifies that consensus even if the
consensus is wrong, sacrificing the diversification benefit that
motivates the multi-agent design in the first place.

## 4.2 Why structure-aware aggregation is needed

Two empirical patterns suggest that the aggregation problem has
exploitable structure rather than being irreducibly noisy.

Agent Shapley values vary across volatility regimes in ways that are
heterogeneous across agents.
Some agents contribute more consistently in low-volatility periods,
others in elevated-volatility regimes, and the geopolitical agent's
average harm is concentrated in specific periods.
This regime-conditional variation means the optimal aggregation
weights are time-varying.
A method that conditions on market state and on each agent's recent
contribution history can in principle recover the weights that static
confidence-weighting cannot.

The debate influence graph $G^{\text{dbt}}_t$ defined in Section 6.2
offers a second source of structure.
$G^{\text{dbt}}_t$ records, for each day $t$, which agents revised
their adjustments in the direction of other agents' Round 1 positions.
The density of this graph is positively associated with the
herding count on that day ($r = 0.37$): sessions in which many
influence edges are active tend to be sessions in which many agents
herd.
An aggregation method that can read the communication structure and
down-weight agents whose revisions were strongly shaped by others
would recover some of the diversity that herding erodes.
A static rule operating on confidence scores alone has no access to
this information.

It is important to note that these two patterns are descriptive
rather than causal.
After controlling for the contemporaneous volatility regime,
neither communication density nor herding count significantly
predicts forecast error in isolation.
The patterns do not prescribe a specific functional form for the
aggregator; they establish that the relevant conditioning variables
exist and are observable at time $t$, motivating a learned approach
that can discover which combinations of those variables are
predictively useful.

[Figure 2 here: two panels.
Panel A shows mean Shapley value by agent and volatility regime,
illustrating that agent contributions are heterogeneous across
regimes and that no agent dominates uniformly.
Panel B shows a scatter of daily influence-edge count in
$G^{\text{dbt}}_t$ against the daily herding count,
illustrating the positive association between communication
density and conformist behaviour.]

## 4.3 Why sparsity is desirable

The observations above point toward a learned, graph-structured
aggregator, but they do not yet explain why sparsity should be a
design objective.

When agents herd, their Round 2 outputs contain duplicated
information: several agents have effectively converged to the same
signal, and a combination that weights them as if they were independent
overcounts that signal.
A fully connected aggregation graph, in which every agent can
influence every other, provides no mechanism for discounting
this redundancy.
If the network learns to propagate information along all edges,
the herded consensus can flow back and forth across the graph,
amplifying rather than correcting for the reduction in effective
diversity.

Sparse aggregation provides a structural remedy.
By restricting the graph to a small number of active edges, the
network is forced to identify which agent-to-agent information
flows are genuinely predictive and to ignore the rest.
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
