# Section 5 Review: Round 2

## Summary Assessment

The revision addresses all three major issues from Round 1 and most minor issues. The context vector's 8th component is now correctly identified as the five-day change. The softmax notation has been replaced with an explicit fraction. The length has been reduced from approximately 2500 words to approximately 1560, which is well within the 2-page target. The three-layer sparsity narrative remains intact and clearly presented. The writing is clean, notation is consistent, and the section reads as a self-contained architecture description suitable for IJF. Two minor issues remain, neither of which threatens convergence.

## Score: 4.38/5.0

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| venue_conformity | 0.18 | 4.5 | 0.81 |
| argument_clarity | 0.16 | 4.5 | 0.72 |
| evidence_support | 0.16 | 4.0 | 0.64 |
| logical_flow | 0.14 | 4.5 | 0.63 |
| project_standards | 0.12 | 4.5 | 0.54 |
| writing_quality | 0.12 | 4.5 | 0.54 |
| statistical_rigor | 0.06 | 4.0 | 0.24 |
| citation_coverage | 0.06 | 4.0 | 0.24 |
| **Total** | **1.00** | | **4.36** |

## Venue Conformity Detail

- Target length: ~2 pages. Actual: ~1560 words, which typesets to approximately 1.8 pages including equations and the figure placeholder. This is within the target.
- Subsection structure (5.1 through 5.6) mirrors the SpotV2Net template. Five numbered equations cover the key architecture components. Figure 1 placement at the section opening is correct.
- Citation style: Parenthetical dominant. Author-as-subject occurrences are limited to one per paragraph. The Lakshminarayanan et al. (2017) citation uses "following" with the parenthetical form, which is acceptable.
- The design-history passages flagged in Round 1 (L1 narrative, concatenation-vs-pooling digression, validation-based early stopping justification) have been trimmed appropriately. Appendix B references are retained for completeness.

## Project Standards Compliance

1. No unnecessary brackets, quotes, arrows: PASS
2. Minimize dash usage: PASS (dashes used sparingly: "regime-gated", "single-agent", etc. are compound modifiers, not stylistic dashes)
3. No "comprehensive": PASS
4. No bold markdown in body text: PASS
5. No sequential markers: PASS
6. Max 1 author-as-subject per paragraph: PASS (checked all paragraphs)
7. Flowing prose, not bullet lists: PASS
8. Model name DropEdge GAT: PASS (used in Section 5.6)

## Consistency Lock Verification

1. Behavior encoding: PASS. "a normalised four-category behaviour code $b_i(t)/3$, where the four categories are herding, anchored, independent, and overconfident, giving values in $\{0, 1/3, 2/3, 1\}$."
2. Context regime: PASS. "a normalised four-level volatility regime indicator $r(t)/3$, where the four levels correspond to persistence thresholds at 0.20, 0.35, and 0.55."
3. Graph embedding: PASS. Concatenation (flatten), graph_dim = 7*16 = 112, combined = 112 + 9 = 121. Text states "121-dimensional combined representation $\mathbf{z}(t)$."
4. Regime gate: PASS. Section 5.5 correctly states inputs as [persist_vol, 1-day change, n_herding/7]. The delta notation is now defined inline: "$\Delta_1\text{rv}_{20,t} \equiv \text{rv}_{20,t} - \text{rv}_{20,t-1}$." This matches the code at `optimized_gat.py` line 468 (`persist_vol.diff().fillna(0)`).
5. Context vector 8th component: PASS. Now reads "the five-day change in persistence volatility $\Delta_5 \text{rv}_{20,t}$." This matches the code at `causal_gat_aggregation.py` line 250 (`persist_vol.diff(5)`).
6. DropEdge: PASS. "20% of active edges are randomly set to zero. Self-loops are exempt. At inference, all learned edges remain active."
7. No prior, variants in Appendix B: PASS. Section 5.2 describes learning from scratch. Granger priors deferred to Appendix B.
8. Training: PASS. Adam lr=0.003, wd=1e-4, cosine LR, 250 epochs, best training loss, grad clip 1.0.
9. Three sparsity layers: PASS. Sigmoid gating (Section 5.2, "first of three"), top-k (Section 5.3, "second sparsity layer"), DropEdge (Section 5.4, "third sparsity layer").

## Round 1 Issue Resolution

### Major Issues

- M1 (context vector 8th component, 5-day vs 1-day): FIXED. Now reads "the five-day change in persistence volatility $\Delta_5 \text{rv}_{20,t}$."
- M2 (length overshoot ~30%): FIXED. Reduced from ~2500 to ~1560 words. The design-history digressions (L1 failure narrative, concatenation-vs-pooling justification, early stopping discussion) have been trimmed. Appendix B cross-references retained.
- M3 (ambiguous softmax notation): FIXED. Equation (3) now uses the explicit fraction $\alpha_{ij}^{(k)} = \exp(e_{ij}^{(k)}) / \sum_{l \in \mathcal{N}_k(j)} \exp(e_{lj}^{(k)})$ with the neighbourhood $\mathcal{N}_k(j)$ defined in the preceding sentence.

### Minor Issues

- m1 (cross-reference for statistical model claim): FIXED. Now reads "as documented in Appendix B."
- m2 (expressiveness claim softened): FIXED. Changed to "with equivalent empirical performance for fixed-size node sets."
- m3 (Lakshminarayanan grammar): FIXED. Now reads "following Lakshminarayanan et al. (2017) for uncertainty reduction through deep ensemble averaging."
- m4 ("uniform probability" to "connection probability"): FIXED. Now reads "placing the initial graph close to a connection probability of 0.5."
- m5 (both heads receive graph embedding): FIXED. Section 5.5 now states "each receive $\mathbf{z}(t)$" and "the regime head trained to specialise on regime-dependent patterns within the shared graph representation."
- m6 (define delta notation for regime gate): FIXED. Inline definition added: "$\Delta_1\text{rv}_{20,t} \equiv \text{rv}_{20,t} - \text{rv}_{20,t-1}$."
- m7 (consistent concatenation notation): FIXED. Multi-head concatenation now uses $\text{concat}_{k=1}^{4}$ rather than the ambiguous vertical-bar notation.

## Minor Issues (should fix)

1. (evidence_support, Section 5.5) The lower-bound clamp at 0.05 is still justified only as "consistent with the range observed in the sample." This is not wrong, but a single parenthetical stating the observed minimum would be more convincing to a replication-minded referee. For example: "consistent with the range observed in the sample, where the minimum 20-day realised volatility exceeds 0.06." This is a polish item, not a blocker.

2. (writing_quality, Section 5.3) Equation (3) is followed by a comma-free transition. The line "$\alpha_{ij}^{(k)} = \ldots$," ends with a comma, then the next line begins "Node $a_j$'s updated representation..." There is a missing period or connecting phrase after the displayed equation block. A period after the equation or a transition such as "The normalised attention coefficient for source $i$ and destination $j$ is" before the equation would improve readability. This is a minor typographic issue.

## Strengths

1. The three-layer sparsity narrative (sigmoid gating, top-k masking, DropEdge) is the most distinctive organisational contribution of this section. It provides a coherent thread that connects Sections 5.2 through 5.4 and gives the IJF reader, who may not be a GNN specialist, a clear mental model of how sparsity emerges at multiple scales.

2. The regime-gated dual-head output (Section 5.5) is presented with full mathematical detail, including the gate input definition, the convex combination formula, and the residual correction. The inline definition of $\Delta_1\text{rv}_{20,t}$ resolves the Round 1 ambiguity cleanly.

3. The length reduction is well executed. The section reads more tightly without sacrificing any essential architectural detail. Design-history material (L1 experiments, causal priors, validation-based early stopping alternatives) is now properly deferred to Appendix B or compressed to single sentences.

4. The training protocol (Section 5.6) is fully reproducible: all hyperparameters, the checkpoint selection rule, the walk-forward embedding, the embargo, and the ensemble protocol are stated with no ambiguity. This is important for IJF, which values replicability.

5. Consistency with the codebase is now verified on all nine lock items. The regime gate inputs (1-day diff for regime_feats vs. 5-day diff in the context vector) are correctly distinguished across Sections 5.1 and 5.5, matching the two separate code paths in `causal_gat_aggregation.py` and `optimized_gat.py`.

## Verdict: CONVERGED

Score 4.36 exceeds the 4.0 balanced threshold. All three major issues from Round 1 are resolved. Two minor polish items remain (clamp justification, punctuation after equation 3), neither of which affects the technical correctness or venue conformity of the section. The section is ready for integration into the manuscript draft.
