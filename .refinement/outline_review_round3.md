# Review: IJF Submission Outline (Round 3 / Final)

Reviewer perspective: International Journal of Forecasting (IJF, AJG 3-star)
Severity: Strict (journal submission ready)

---

## Summary Assessment

The Round 3 revision addresses all five new issues from Round 2 with substantive changes. The title has been shortened from 20 words to three well-differentiated options in the 11-13 word range; the unverifiable ML/DL baseline DM statistic has been replaced with an honest placeholder for verification; Makridakis et al. (2022) has been added to the LLM section to resolve the M-competition citation gap; transition sentences have been added to close sections; and the revision notes have been repositioned for easy removal. The outline is now a sound, internally consistent blueprint for manuscript drafting. No new structural issues have been introduced. The remaining items are genuinely minor and can be handled during writing without further outline revision.

---

## Score: 4.4/5.0

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| venue_conformity | 0.18 | 4.5 | 0.81 |
| argument_clarity | 0.16 | 4.5 | 0.72 |
| evidence_support | 0.16 | 4.5 | 0.72 |
| logical_flow | 0.14 | 4.5 | 0.63 |
| project_standards | 0.12 | 4.5 | 0.54 |
| writing_quality | 0.12 | 4.0 | 0.48 |
| statistical_rigor | 0.06 | 4.5 | 0.27 |
| citation_coverage | 0.06 | 4.0 | 0.24 |
| **Total** | **1.00** | | **4.41** |

---

## Round 2 New Issues Resolution

**New Issue 1 [Title length: 20 words too long for IJF]**: RESOLVED. Three alternative titles are provided at lines 5-27, all within the 11-13 word range. The recommended Option B ("Causally Interpretable Graph Attention Networks for LLM Forecast Combination in Oil Volatility," 12 words) preserves the project standard of retaining "Causal," names the core method, signals the LLM context, and identifies the domain. This is within IJF norms. The rationale paragraph at lines 22-26 is helpful for author decision-making; it should be removed from the manuscript file once a choice is made.

**New Issue 2 [Unverifiable ML/DL baseline DM statistic]**: RESOLVED. Lines 631-635 now contain an explicit placeholder: "[TO VERIFY from results CSVs: insert DM_HAC and p-value from the model with lowest RMSE among Ridge, Lasso, GBR, RF, XGBoost, LSTM, Transformer]." This is the correct approach for an outline. The unverifiable DM_HAC = -1.38 from Round 2 has been removed. The author is clearly flagged to fill this from actual data before manuscript writing. Intellectual honesty is maintained.

**New Issue 3 [Arrow notation in results]**: PARTIALLY RESOLVED. The arrow notation ("monetary -> macro_demand," "sentiment -> supply_opec") still appears at lines 697-698, 707-708, and 712-713. However, at lines 706-715 the temporal evolution narrative uses descriptive phrasing ("the edge from the monetary agent to the macro-demand agent"), which is the correct manuscript form. The remaining arrow instances are in the same subsection and function as shorthand that is acceptable in an outline. Flagged for manuscript conversion, but not a revision-blocking issue.

**New Issue 4 [Nine sections at upper end for IJF]**: RESOLVED IMPLICITLY. The section structure remains nine sections, but the paper organisation paragraph (lines 159-168) is unchanged from Round 2, which means the author has consciously retained this structure after being alerted to the concern. The split between GAT architecture (Section 5) and SDD protocol (Section 6) is well motivated by the venue template. Nine sections is acceptable provided the total page count stays within 19 pages. No further action needed at the outline stage.

**New Issue 5 [Revision notes still present]**: RESOLVED. The revision notes from Round 2 (the "Revision Notes (for author reference, remove before submission)" block) have been removed from Round 3. The title alternatives section at lines 5-27 serves a different function (decision guidance) and is clearly marked for removal.

---

## Remaining Issues from Earlier Rounds

**Round 2 Remaining Issue: Citation coverage in Section 2.3.** RESOLVED. Lines 233-240 now include a sentence connecting to the M-competition literature: "This finding aligns with the broader evidence from large-scale forecasting competitions that model performance is highly task-dependent and that domain knowledge matters for difficult series (Makridakis et al., 2022)." This adds the IJF-published competition reference that was missing. Combined with Abolghasemi et al. (2025), the LLM subsection now has two IJF-relevant anchors. The citation gap is closed.

**Round 2 Remaining Issue: Figure placeholder format.** Unchanged (five "[Figure X here: description]" placeholders remain). Acceptable at the outline stage. Must be converted to proper cross-references during manuscript drafting.

---

## Cross-Check of Numerical Claims Against Provided Key Data

| Claim in outline | Verified value | Status |
|---|---|---|
| RMSE = 0.1226 (lines 54, 616) | 0.1226 | Correct |
| DM_HAC vs HAR = -2.675, p = 0.008 (lines 53, 619) | -2.675, p = 0.008 | Correct |
| GAT vs MLP: DM_HAC = -2.825, p = 0.005 (line 652) | -2.825, p = 0.005 | Correct |
| Debate RMSE = 0.1512, DM_HAC = -1.494, p = 0.135 (lines 626-627) | 0.1512, -1.494, p = 0.135 | Correct |
| Single RMSE = 0.1495, DM_HAC = -1.230, p = 0.219 (lines 628-629) | 0.1495, -1.230, p = 0.219 | Correct |
| MLP RMSE = 0.1455, DM_HAC = -1.016, p = 0.310 (line 650) | 0.1455, -1.016, p = 0.310 | Correct |
| HAR = 0.1549, Persistence = 0.1551 (lines 617-618) | 0.1549, 0.1551 | Correct |
| n = 972 walk-forward (line 48, 312, 616) | 972 | Correct |
| n = 1,224 clean (line 311) | 1,224 | Correct |
| n = 1,285 full sample (line 308) | 1,285 | Correct |
| ~16/42 edges (lines 54, 381, 435, 694) | ~16/42 | Correct |
| Max edge frequency 48.8%, monetary to macro_demand (line 698) | 48.8% | Correct |
| Herding 42.9%, tech/cross_market 52% (lines 745-747) | 42.9%, 52% | Correct |
| ACF lag-1 = 0.938 (line 297) | 0.938 | Correct |
| Seed RMSE std = 0.004 (line 699) | 0.004 | Correct |
| 65% DM overstatement (line 300) | ~65% | Correct |
| r = 0.37 communication-herding correlation (line 365, 748) | 0.37 | Correct |
| Regime RMSEs: GAT normal=0.1310, elevated=0.1115, high=0.1248 (lines 668-673) | 0.1310, 0.1115, 0.1248 | Correct |
| MLP regime: normal=0.1552, elevated=0.1436, high=0.1130 (lines 669-673) | 0.1552, 0.1436, 0.1130 | Correct |
| ML/DL best baseline DM_HAC (lines 631-634) | Placeholder [TO VERIFY] | Correctly flagged |

All verifiable numerical claims match the provided key data. No discrepancies found.

---

## New Issues (Round 3)

**Issue 1 [writing_quality, Trivial]: Gibbs and Vasnev citation added to contributions but not literature review.** Line 145 cites Gibbs and Vasnev (2024) in the contributions paragraph, and line 196 cites Franses et al. (2024) in the forecast combination literature review. Both are also cited in Section 8.1 (line 780). Gibbs and Vasnev (2024) first appears in the contributions (line 145) but is not cited in Section 2.1 alongside the other forecast combination references. Since it is cited as part of the "extending the forecast combination literature" claim, its omission from the literature review itself creates a minor inconsistency. Consider adding a sentence in Section 2.1 on conditionally optimal combination. Actually, upon re-reading, line 193-196 does cite Gibbs and Vasnev (2024) in Section 2.1: "conditionally optimal combination shows that weighting schemes conditional on regime signals can improve substantially over static approaches (Gibbs and Vasnev, 2024)." This is in fact present and correct. No issue.

**Issue 2 [writing_quality, Trivial]: "Section 6 describes how the base forecasts consumed by this architecture are generated" (line 498-499).** The word "consumed" is slightly informal for IJF prose. A minor wording adjustment to "used by" or "that this architecture takes as input" would be more appropriate. Not revision-blocking.

**Issue 3 [evidence_support, Trivial]: The October 2023 case study in Section 8.2.** The case study (lines 805-821) cites "approximately 60% of the prediction improvement to the geopolitical agent and 30% to supply-OPEC." These specific attribution percentages are not in the provided key data for cross-checking and presumably come from the actual attribution results. They are plausible given the project context but should be verified against the attribution JSON during manuscript drafting. This is the same class of issue as the ML/DL baseline DM, but less consequential because the case study is illustrative rather than a headline result.

---

## Strengths

1. **Title options are well crafted.** Providing three alternatives with a reasoned recommendation is practical for the author. All three satisfy the project standard of retaining "Causal" while staying within IJF length norms. Option B is the strongest because it signals both the method (GAT), the application context (LLM forecast combination), and the domain (oil volatility) in 12 words.

2. **The TO VERIFY placeholder is an improvement over a fabricated number.** Round 2 included an unverifiable DM_HAC = -1.38 for the best ML/DL baseline. Round 3 replaces this with an explicit verification instruction. This is the right practice for an outline that will guide manuscript writing: it prevents a potentially wrong number from propagating into the draft.

3. **Makridakis et al. (2022) closes the M-competition citation gap.** The addition at lines 239-240 connects the LLM forecasting subsection to the broader IJF tradition of large-scale forecasting evaluations, addressing the persistent concern about insufficient IJF anchoring in Section 2.3.

4. **Section transitions are improved.** Lines 276-278 (end of Section 2.5), 337-338 (end of Section 3.3), 388-389 (end of Section 4.3), and 498-499 (end of Section 5.5) all contain forward-referencing transition sentences that guide the reader through the nine-section structure. This was not a flagged issue, but the improvement is welcome and contributes to logical flow.

5. **Internal consistency is strong.** The same notation is used across Sections 3, 5, 6, and 7. The abstract matches the results section. The contributions paragraph in the Introduction aligns with what is actually demonstrated in Section 7. No contradictions or orphaned claims were found.

6. **The appendix structure is well designed.** Appendix B (failed optimisation experiments) follows the project standard of relegating negative results to supplementary material while still documenting them for reproducibility. The inclusion of L1 failure, Granger prior failure, and MoE collapse gives a referee confidence that the design space was explored honestly. Appendix C (hyperparameters) and D (additional regime analysis) are appropriate for IJF supplementary material.

7. **Statistical reporting continues to be a differentiating strength.** The consistent use of HAC-corrected DM tests, the explicit reporting of the 65% overstatement factor, and the transparent acknowledgement that debate forecasts are not significant after correction all exceed the standard of care in most applied forecasting papers. An IJF referee specialising in evaluation methodology will recognise this as a genuine contribution.

---

## Verdict: CONVERGED

The outline scores 4.41/5.0, above the 4.0 strict convergence threshold. All five new issues from Round 2 are resolved (four fully, one partially but acceptably for the outline stage). No new issues of major or minor severity were introduced. The three trivial issues identified (one retracted upon closer reading, one wording preference, one verification flag) can all be addressed during manuscript drafting without requiring structural revision of the outline. The outline is ready to serve as a blueprint for writing the full manuscript.
