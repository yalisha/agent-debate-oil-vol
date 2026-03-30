# Section 5 Review: Round 1

## Summary Assessment

This is a strong first draft that presents the DropEdge GAT architecture with appropriate mathematical detail and clear logical progression from inputs through graph construction, attention layers, DropEdge, and dual-head output. The writing is mostly clean and the consistency corrections from the outline are properly implemented. However, the section runs approximately 30% over the target length, contains one factual error in the context vector description (5-day vs 1-day volatility change), and several passages provide design-history narration that belongs in an appendix rather than the main method section.

## Score: 3.72/5.0

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| venue_conformity | 0.18 | 3.0 | 0.54 |
| argument_clarity | 0.16 | 4.0 | 0.64 |
| evidence_support | 0.16 | 3.5 | 0.56 |
| logical_flow | 0.14 | 4.0 | 0.56 |
| project_standards | 0.12 | 4.5 | 0.54 |
| writing_quality | 0.12 | 4.0 | 0.48 |
| statistical_rigor | 0.06 | 3.0 | 0.18 |
| citation_coverage | 0.06 | 3.5 | 0.21 |
| **Total** | **1.00** | | **3.71** |

## Venue Conformity Detail

- Target length: ~2 pages vs actual: ~3.2 pages (approx. 2500 words) -> TOO LONG by ~30%
- Structure match: Subsection organisation (5.1 through 5.6) is appropriate and mirrors the SpotV2Net template. Full equations for attention mechanism, final prediction, and regime gating are present as expected. The figure placement at the top is correct.
- Citation style: Parenthetical dominant. Two author-as-subject occurrences: "Velichkovic et al. (2018)" and "Rong et al. (2020)". Both are within acceptable limits (max 2 per paragraph). Lakshminarayanan et al. (2017) and Kingma and Ba (2015) are parenthetical. Acceptable.
- Deviation: The section is too long for IJF standards, which explicitly caution that "Referees will be asked to consider the value of the paper relative to its length." Several passages explain what was tried and failed (L1 regularisation, validation-based early stopping, causal priors), which constitutes results-flavoured design history that would be more appropriate in Appendix B. Trimming these would bring the section within the 2-page target.

## Project Standards Compliance

1. No unnecessary brackets, quotes, arrows: PASS
2. Minimize dash usage: PASS (dashes used sparingly and appropriately)
3. No "comprehensive": PASS
4. No bold markdown in body text: PASS
5. No sequential markers: PASS (no "first, second, third" enumeration)
6. Max 1 author-as-subject per paragraph: PASS (checked all paragraphs)
7. Flowing prose, not bullet lists: PASS
8. No label-style headings: PASS
9. Framework name SDD: PASS (used where appropriate)
10. Model name DropEdge GAT: PASS

## Consistency Lock Verification

1. **Behavior encoding**: PASS. Text correctly states "a normalized four-category behaviour code $b_i(t)/3$, where the four categories are herding, anchored, independent, and overconfident, giving values in $\{0, 1/3, 2/3, 1\}$."
2. **Context regime**: PASS. Text correctly states "a normalized four-level volatility regime indicator $r(t)/3$, where the four levels correspond to persistence thresholds at 0.20, 0.35, and 0.55."
3. **Graph embedding**: PASS. Text correctly states "the seven node representations are concatenated rather than pooled, yielding a graph embedding of dimension $7 \times 16 = 112$" and "$\mathbf{z}(t) = [\text{flatten}(\mathbf{H}') \| \mathbf{c}(t)]$."
4. **Regime gate**: PASS. Text correctly states the three inputs as "the persistence volatility $\text{rv}_{20,t}$, the one-day change in persistence volatility $\Delta\text{rv}_{20,t}$, and the normalized herding count $n_{\text{herd}}(t)/7$." This matches the code (`persist_vol.diff()` = 1-day diff) and the consistency lock.
5. **DropEdge**: PASS. "20% of active edges are randomly set to zero before the GAT layers compute attention. Self-loops are exempt from this dropout. At inference, all learned edges remain active."
6. **No prior in final model**: PASS. Section 5.2 describes learning from scratch, and causal priors are correctly deferred to Appendix B.
7. **Training protocol**: PASS. Adam lr=0.003, wd=1e-4, cosine LR, 250 epochs, best training loss checkpoint, grad clip 1.0 all match.
8. **Three sparsity layers**: PASS. Explicitly labelled as "the first of three sparsity layers" (Section 5.2, sigmoid gating), "the second sparsity layer" (Section 5.3, top-k), and "the third sparsity layer" (Section 5.4, DropEdge).

## Major Issues (must fix)

1. **statistical_rigor / venue_conformity** at Section 5.1 (context vector, 8th component): The text says "the one-day change in persistence volatility" but the code in `build_features()` computes `df['vol_change_5d'] = df['persist_vol'].diff(5).fillna(0)`, which is a 5-day change. The 1-day change is used only in the regime gate features, not in the context vector. This is a factual error that could mislead a reader attempting to replicate the model. Fix: change "the one-day change in persistence volatility" to "the five-day change in persistence volatility $\Delta_5 \text{rv}_{20,t}$" in the context vector description. (Note: the regime gate description in Section 5.5 correctly states 1-day, which is consistent with the code. The problem is only in Section 5.1.)

2. **venue_conformity** (length): At approximately 2500 words / 3.2 typeset pages, the section exceeds the ~2-page target by roughly 30%. IJF referees are sensitive to length. The following passages are candidates for cutting or moving to Appendix B:
   - Section 5.2, final paragraph ("This mechanism constitutes the first of three sparsity layers... suggesting that the data-driven topology is not simply recovering a known causal structure.") This is partially results discussion. Retain the one-sentence summary that no prior is used; move the causal-prior experiment to Appendix B.
   - Section 5.4, the entire second paragraph ("DropEdge also resolves a design tension..."). This narrates the L1 failure, which is design history and already referenced via Appendix B. A single sentence noting that DropEdge is preferred over L1 regularisation (Appendix B) suffices.
   - Section 5.6, the parenthetical about validation-based early stopping. One clause is sufficient.
   - Section 5.3, the passage about concatenation vs mean-pooling justification (three sentences). This is a reasonable design choice explanation but could be condensed to one sentence.

3. **argument_clarity** at Section 5.3 (attention equation notation): The notation $\alpha_{ij}^{(k)} = \text{softmax}_{i}(e_{ij}^{(k)})$ is potentially ambiguous. The subscript $i$ on the softmax could mean "softmax computed at node $i$" or "softmax over index $i$." Since $i$ is the source and $j$ is the destination, and the normalisation is over sources for each destination, the intended meaning is "softmax over index $i$ for fixed $j$." Standard notation in the GAT literature writes this as $\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}(j)} \exp(e_{lj}^{(k)})}$ or uses $\text{softmax}_{j \leftarrow \cdot}$. Fix: replace the ambiguous subscript notation with the explicit fraction form, or write $\text{softmax}_{i \in \mathcal{N}(j)}$.

## Minor Issues (should fix)

1. **writing_quality** at Section 5.1 (node feature list): The seven components are presented as a single long sentence with semicolons separating them. At seven items this is taxing to parse. Consider presenting this as a compact inline enumeration or a small table (consistent with SpotV2Net's approach of tabling feature definitions). The same applies to the nine context vector components.

2. **evidence_support** at Section 5.1: The claim that "adding statistical model outputs (HAR, GARCH, persistence) as additional base forecasts reduces accuracy" is stated without a cross-reference. This finding appears in the ablation results. Add a forward reference: "as shown in Section 7" or "Appendix B."

3. **writing_quality** at Section 5.3: The sentence "The decomposed additive formulation avoids the full concatenation of the original GAT paper (Velickovic et al., 2018) at no loss in expressive power for fixed-size node sets" makes an expressiveness claim without justification. Either cite a reference that proves equivalence for fixed-size graphs, or soften to "with equivalent empirical performance."

4. **citation_coverage** at Section 5.4: The DropEdge citation "Rong et al., 2020" should be verified for the correct year. The original DropEdge paper was published at ICLR 2020, so the year is correct, but the reference list should include the full venue (ICLR).

5. **statistical_rigor** at Section 5.5 (equation 5): The lower-bound clamp at 0.05 is described as "consistent with the range observed in the sample." This is an empirical statement. It would be more rigorous to state the observed minimum annualised volatility in the sample (e.g., "the minimum 20-day realised volatility in the clean sample is X, so the floor at 0.05 is non-binding in practice").

6. **logical_flow** at Section 5.5: The regime gate description introduces "the persistence volatility $\text{rv}_{20,t}$, the one-day change in persistence volatility $\Delta\text{rv}_{20,t}$, and the normalized herding count $n_{\text{herd}}(t)/7$." These symbols have been defined earlier in the context vector (Section 5.1), but the delta notation $\Delta\text{rv}_{20,t}$ appears for the first time here. Define it inline or use a consistent symbol that was introduced in Section 5.1.

7. **writing_quality** at Section 5.6: "following the practice of (Lakshminarayanan et al., 2017)" has awkward grammar. Change to "following Lakshminarayanan et al. (2017)" or, if author-as-subject limit is a concern, rephrase to "following the deep ensemble averaging practice established in the literature (Lakshminarayanan et al., 2017)."

8. **statistical_rigor** at Section 5.2: "The logits are initialised by drawing from $\mathcal{N}(0, 0.1^2)$, placing the initial graph close to uniform probability (0.5) for every possible edge." This is correct (sigmoid(0)=0.5 and logits drawn near 0), but the phrase "close to uniform probability" could be misread as "uniform distribution." Consider "close to a connection probability of 0.5."

9. **argument_clarity** at Section 5.5: The sentence "When $g(t)$ is near zero the base head dominates, producing a combination that is primarily driven by the agent-interaction patterns encoded in the graph" conflates the base head with the graph. Both heads receive the graph embedding $\mathbf{z}(t)$ as input; the distinction is that the regime head's parameters are trained to be regime-sensitive. Clarify that both heads access graph information, and the regime head is intended to specialise on regime-dependent patterns.

10. **notation consistency** at Section 5.3: The node update equation uses $\bigl\|_{k=1}^{4}$ for concatenation, while Section 5.1 uses $\|$ for vector concatenation in $\mathbf{z}(t)$. Both are standard, but the vertical bar with subscript/superscript may render ambiguously in some typesetters. Consider using $\oplus$ or $\text{concat}_{k=1}^{4}$ for the multi-head concatenation.

## Strengths

1. The three-layer sparsity narrative (sigmoid gating, top-k masking, DropEdge) provides a coherent and original organisational thread that ties the architectural choices together. This is well suited to the IJF readership, who may not be familiar with GNN design patterns.

2. The consistency corrections from the outline are all correctly implemented. The behavior encoding, context regime, graph embedding, and regime gate inputs now match the code exactly (except for the context vector's 5-day vs 1-day issue noted above).

3. The decision to concatenate rather than pool node representations is well motivated by the semantic distinctness of the seven agents. This justification is appropriate for a forecasting audience that would otherwise expect standard pooling.

4. The training protocol description is precise and reproducible: all hyperparameters, the checkpoint selection rule, the walk-forward embedding, and the ensemble protocol are clearly stated with no ambiguity.

5. The section correctly defers causal-prior experiments to Appendix B and results discussion to Section 7, maintaining a clean separation between method description and empirical findings.

## Verdict: NEEDS_REVISION

The section is well structured and mostly accurate, but three issues prevent convergence: (a) the factual error in the context vector's 8th component (5-day, not 1-day change), (b) the ~30% length overshoot relative to the IJF target, and (c) the ambiguous softmax subscript notation. Fixing these should bring the score above 4.0.
