# Review: IJF Submission Outline (Round 2)

Reviewer perspective: International Journal of Forecasting (IJF, AJG 3-star)
Severity: Strict (journal submission ready)

---

## Summary Assessment

The Round 2 revision addresses all five major issues from Round 1 with genuine structural changes rather than cosmetic fixes. The title has been reworked to position "causal" as an interpretive claim rather than a methodological one; the methodology has been split into separate GAT architecture and SDD protocol sections following the venue template; the context vector is properly introduced before the components that consume it; and an architecture diagram is now the primary figure. The writing quality has improved throughout, with scholar-as-subject constructions reduced, own-data statistics removed from the literature review, and the third contribution reframed as an empirical demonstration rather than a methodological claim. The outline is now structurally sound for IJF submission. Remaining issues are minor and concern the title length, a few residual writing patterns, and small gaps in citation coverage.

---

## Score: 4.2/5.0

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| venue_conformity | 0.18 | 4.0 | 0.72 |
| argument_clarity | 0.16 | 4.5 | 0.72 |
| evidence_support | 0.16 | 4.0 | 0.64 |
| logical_flow | 0.14 | 4.5 | 0.63 |
| project_standards | 0.12 | 4.0 | 0.48 |
| writing_quality | 0.12 | 4.0 | 0.48 |
| statistical_rigor | 0.06 | 4.5 | 0.27 |
| citation_coverage | 0.06 | 3.5 | 0.21 |
| **Total** | **1.00** | | **4.15** |

---

## Round 1 Issues Resolution

### Major Issues

**Major 1 [Title: "Causal" misleading]**: RESOLVED. The new title reads "Graph Attention Networks with Causally Interpretable Learned Structure for Forecast Combination." The word "Causal" now modifies "Interpretable Learned Structure," making clear that causality is an interpretive lens applied to a data-driven graph, not the graph construction method. A referee will understand from this title that the graph is learned. The rationale note (lines 26-31) is helpful for the author but should be removed before submission.

**Major 2 [Methodology section structure]**: RESOLVED. The former Section 5 has been split into Section 5 (GAT Architecture, ~2 pages) and Section 6 (Structured Delphi Debate Protocol, ~1.5-2 pages). This matches the venue template's separation of model architecture from domain-specific methodology, and correctly places the core contribution (GAT) before the application design (SDD).

**Major 3 [ML/DL baselines dismissed without statistics]**: RESOLVED. Section 7.2 (lines 601-604) now reports "the best-performing model achieves DM_HAC = -1.38 (p = 0.168) against HAR, failing to reach conventional significance thresholds after HAC correction; the full statistics for all 13 models are in Table 1." This is the right approach: one sentence of context plus a table reference.

**Major 4 [Context vector backward reference]**: RESOLVED. The context vector is now introduced in Section 5.1 (lines 395-403), before any component that uses it. The regime gate description in Section 5.4 can now forward-reference the context vector without confusion.

**Major 5 [No architecture diagram]**: RESOLVED. Figure 1 is now the two-stage architecture diagram (lines 375-380), with the herding heatmap demoted to Figure 2. The figures plan (lines 905-910) includes proper annotation guidance for the architecture diagram.

### Minor Issues

**Minor 1 [Own-data herding stats in Lit Review]**: RESOLVED. Section 2.5 now contains only published literature. The 42.9% and 52% herding statistics appear in Section 7.6 (lines 711-713) where they belong.

**Minor 2 [Numeric results in motivation section]**: RESOLVED. Section 4.1 (lines 329-331) now reads "Preliminary analysis shows that the confidence-weighted mean does not significantly outperform HAR after correcting for overlapping-horizon autocorrelation (results in Section 7)." No specific numbers are reported in the motivation section.

**Minor 3 ["Include Figure X here" formatting]**: PARTIALLY RESOLVED. The format has changed from "Include Figure X here:" to bracketed "[Figure X here: description]" format, which is marginally better for an outline stage. Five such placeholders remain. Acceptable at the outline stage; must be converted to proper figure references in the manuscript.

**Minor 4 [Third contribution reframed]**: RESOLVED. Lines 149-154 now read: "it demonstrates that proper treatment of overlapping-horizon autocorrelation reverses the apparent significance of naive LLM debate forecasts...establishing that the learned combination mechanism, not the quality of the base forecasts alone, is the source of genuine forecasting gains." This is precisely the reframing suggested.

**Minor 5 [Missing combination literature citations]**: RESOLVED. Smith and Wallis (2009) and Petropoulos et al. (2022) are now cited in both the Introduction (line 85-86) and Section 2.1 (lines 182, 187-189).

**Minor 6 [More IJF citations for LLM section]**: PARTIALLY RESOLVED. No additional IJF citations have been added to Section 2.3. Abolghasemi et al. (2025) remains the only IJF reference in the LLM subsection. The M-competition literature and any IJF-published ML forecasting evaluations are still absent.

**Minor 7 ["Five categories" counting error]**: RESOLVED. Section 7.1 (line 568) now reads "Five baseline categories are included," and the proposed method is listed separately (line 577).

**Minor 8 [Abstract ambiguity: n=1224 vs n=972]**: RESOLVED. The abstract (line 53) now reads "972 out-of-sample predictions" instead of "1,224 trading days," which is the correct figure for the walk-forward evaluation sample.

**Minor 9 [Scholar-as-subject overuse]**: RESOLVED. Section 2.1 has been substantially rewritten. The Wang et al. (2023) citation is now parenthetical: "Three mechanisms by which combination generates value have been identified...variance reduction under independence, bias correction through diversity, and regime adaptation...(Wang et al., 2023)." Section 2.2 uses parenthetical form for Li and Tang (2024). Section 2.3 still uses one subject-position construction for Du et al. (2024) but this is within acceptable limits.

**Minor 10 ["Catfish paper" nickname]**: RESOLVED. Line 110 now reads simply "(Wang et al., 2025)."

**Minor 11 [Seed RMSE variance to support multiple optima claim]**: RESOLVED. Lines 667-670 now report "The 5-seed ensemble exhibits RMSE standard deviation of 0.004 across seeds despite topologically distinct graphs, supporting the interpretation that multiple locally optimal sparse graphs exist."

**Minor 12 [Unsupported claim: SDD maps expert practice]**: RESOLVED. Line 498-499 now reads "This design adapts the Delphi elicitation method to a reproducible multi-agent LLM inference procedure (Dalkey and Helmer, 1963)," which is a defensible statement with a citation.

**Minor 13 [Discussion section expansion]**: RESOLVED. Section 8.2 has been expanded with a concrete worked example (the October 2023 Middle East escalation case study, lines 768-784), showing how a risk manager would interpret the Shapley decomposition on a specific date. This is exactly the kind of practical illustration an IJF referee expects. Limitations are consolidated as Section 8.3 rather than fragmenting the discussion.

**Minor 14 ["Aggregation bottleneck" slogan tone]**: RESOLVED. Line 114-115 now reads "This aggregation problem means that the forecasting value created by diverse LLM agents is largely dissipated by naive combination." More measured phrasing, consistent with IJF tone.

---

## New Issues

**New Issue 1 [venue_conformity, Minor]: Title length.** The new title is 20 words: "Graph Attention Networks with Causally Interpretable Learned Structure for Forecast Combination: Structured Delphi Debate and Sparse Meta-Aggregation in Oil Volatility Forecasting." This is long by IJF standards, where titles typically run 10-15 words. SpotV2Net's title is 13 words. The subtitle after the colon essentially repeats information already conveyed in the main clause. Consider shortening to something like "Learned Sparse Graph Attention for Forecast Combination in Oil Volatility" (10 words) while noting the causal interpretation in the abstract and introduction rather than the title.

**New Issue 2 [evidence_support, Minor]: Best ML/DL baseline DM statistic unverifiable.** Section 7.2 reports "the best-performing model achieves DM_HAC = -1.38 (p = 0.168)" but this specific statistic does not appear in the project's CLAUDE.md data or the verified key numbers. The known DM_HAC values are for debate (-1.494), single agent (-1.230), and MLP (-1.016). The DM_HAC = -1.38 for the best ML/DL baseline is presumably from the actual results files but was not provided for cross-checking. This is not necessarily wrong, but it should be verified against the results CSVs before manuscript writing.

**New Issue 3 [writing_quality, Minor]: Residual "arrow" notation in results.** Lines 666, 675-676, and 679 use the arrow notation "monetary -> macro_demand" and "sentiment -> supply_opec." The project standards specify no unnecessary arrows. In the manuscript, these should be rendered as proper directed edge notation (e.g., "the monetary-to-macro_demand edge" or using mathematical notation with a directed arrow symbol in equations). At the outline stage this is acceptable shorthand, but flag for manuscript conversion.

**New Issue 4 [logical_flow, Minor]: Section numbering gap.** The paper organisation paragraph (lines 160-170) lists nine sections, which is one more than the Round 1 outline. This is reasonable given the methodology split, but the Discussion is now Section 8 and the Conclusion is Section 9. Nine numbered sections is at the upper end for IJF. SpotV2Net uses eight. This is not a problem in itself, but the author should confirm that the split does not push the paper over the length target.

**New Issue 5 [writing_quality, Minor]: Revision notes present.** Lines 2-16 contain "Revision Notes (for author reference, remove before submission)." This is fine for the current stage, but flagged as a reminder.

---

## Remaining Minor Issues

1. **Citation coverage gap in Section 2.3.** The LLM forecasting subsection still lacks additional IJF citations beyond Abolghasemi et al. (2025). The M-competition papers (Makridakis et al., 2022) that now include ML/DL benchmarks would strengthen the IJF anchoring. This was flagged as Minor 6 in Round 1 and is only partially resolved.

2. **Figure placeholder format.** Five "[Figure X here: description]" placeholders remain. These must become proper cross-references in the manuscript but are acceptable at the outline stage.

3. **The DM_HAC = -1.38 for the best ML/DL model needs verification.** See New Issue 2 above.

---

## Strengths

1. **Clean resolution of the title problem.** The phrase "Causally Interpretable Learned Structure" is intellectually honest: it signals that the graph is learned (not pre-specified) and that causality is an interpretation (not the construction method). This simultaneously satisfies the project standard of retaining "Causal" and defuses the Round 1 concern about misleading a referee.

2. **Structural alignment with the venue template is now strong.** Sections 5 and 6 separately address model architecture and domain-specific methodology, matching the SpotV2Net template. The separate notation section, motivation section, and nine-section layout are all defensible for IJF.

3. **The concrete risk management example in Section 8.2 is a genuine improvement.** The October 2023 Middle East escalation case study (lines 768-784) transforms the interpretability discussion from abstract claims into a specific, date-anchored scenario that an energy practitioner or a referee can evaluate. This is the kind of "practical implications" content that the IJF editorial board values.

4. **The context vector relocation eliminates the backward-reference problem entirely.** Section 5.1 introduces both node features and the context vector before any architecture component references them. The logical flow through Sections 5.1 through 5.5 is now linear.

5. **All numerical claims cross-check correctly against the project data.** RMSE values, DM statistics, p-values, sample sizes, edge counts, herding rates, and ACF values are all consistent with the verified key numbers. The only exception is the ML/DL baseline DM_HAC = -1.38, which is plausible but not in the provided cross-check data.

6. **Honest statistical reporting is maintained and strengthened.** The expanded treatment in Section 7.2 (lines 607-610) and Section 8.1 (lines 746-754) of the naive-vs-HAC DM distinction is a differentiating strength for an IJF submission, where methodological rigour in evaluation protocol is weighed heavily.

---

## Verdict: CONVERGED

The outline scores 4.15/5.0, above the 4.0 strict convergence threshold. All five major issues from Round 1 are resolved. No new major issues were introduced. The remaining issues are minor and concern title length, one unverifiable statistic, and a small citation gap. These can be addressed during manuscript drafting without requiring another structural revision of the outline.
