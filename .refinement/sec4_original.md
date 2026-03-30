# Section 4 Original (from outline_ijf.md lines 336-385)
# Saved as backup before refinement

## 4. Motivation for the GAT Architecture (~1 page)

### 4.1 Why naive combination fails

Preliminary analysis shows that the confidence-weighted mean does
not significantly outperform HAR after correcting for
overlapping-horizon autocorrelation (results in Section 7).
This failure cannot be attributed to poor base forecast quality:
each agent individually improves the forecast relative to
persistence, as confirmed by the sign of all Shapley values.
The problem is that herding concentrates forecast mass toward the
consensus, reducing the effective diversity of the combination and
making naive weighting an unreliable aggregation rule.

### 4.2 Empirical motivation for graph structure

Two empirical patterns motivate a graph-based combination.
First, agent Shapley values are not uniformly distributed: some
agents contribute more in specific volatility regimes, suggesting
that the optimal combination structure is time-varying and
potentially discoverable from historical data.
Second, the influence graph constructed from Round 1 to Round 2
adjustment changes exhibits a heterogeneous communication structure.
Communication density correlates with herding rates (r = 0.37),
meaning that more connected networks produce more conformist
behaviour.
A combination method that can learn to discount heavily connected
agents in high-herding periods should outperform naive weighting.

[Figure 2 here: a heatmap of pairwise agent herding rates across
the sample, motivating the need for structured combination.]

### 4.3 Why sparsity matters

A fully connected GAT would allow every agent to influence every
other, potentially reinforcing the herding dynamic that naive
combination fails to correct.
By allowing the network to gate edges via learnable sigmoid logits,
the model discovers the sparse interaction structure that is
predictively useful.
Empirically, the learned graph retains approximately 16 of 42
possible edges across seeds and windows, a sparsity level that is
not imposed by regularisation but emerges from the walk-forward
optimisation.
Appendix B shows that both L1 regularisation and a
Granger-causality prior fail to produce useful sparsity, whereas
the learned logit approach converges reliably.
The architecture that implements this idea is described in the
following section.
