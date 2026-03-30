# Review: IJF Submission Outline (Round 1)

Reviewer perspective: International Journal of Forecasting (IJF, AJG 3-star)
Severity: Strict (journal submission ready)

---

## Summary Assessment

The Round 1 revision is a substantial improvement over the original ESWA outline. The reframing toward forecast combination as the central contribution, the adoption of IJF section structure (separate notation, motivation, and methodology sections), and the honest treatment of HAC-corrected statistical significance all represent the right strategic choices. However, several structural problems remain: the title still uses "Causal" despite the paper demonstrating that causal priors do not work, the Methodology section conflates the two-stage framework into a single section rather than separating domain-specific methodology from model architecture as the venue template requires, and some subsections still read as research notebook entries rather than journal prose. The writing is generally clean but exhibits recurring patterns that an IJF referee would flag.

---

## Score: 3.4/5.0

| Criterion | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| venue_conformity | 0.18 | 3.0 | 0.54 |
| argument_clarity | 0.16 | 4.0 | 0.64 |
| evidence_support | 0.16 | 3.5 | 0.56 |
| logical_flow | 0.14 | 3.5 | 0.49 |
| project_standards | 0.12 | 3.0 | 0.36 |
| writing_quality | 0.12 | 3.0 | 0.36 |
| statistical_rigor | 0.06 | 4.5 | 0.27 |
| citation_coverage | 0.06 | 3.0 | 0.18 |
| **Total** | **1.00** | | **3.40** |

---

## Venue Conformity Detail

- **Target length**: SpotV2Net = 19 pages (~8000-9000 words). The outline is ~5,300 words of structured prose notes, which would expand to roughly 9,500-10,500 words when fully written (accounting for table/figure descriptions becoming actual tables and figures, and notes like "Include Figure 2 here" becoming embedded objects). This is 10-17% over target. The four appendices would add further length. **At the upper edge of acceptable; the appendices may need to go to supplementary material rather than the main manuscript.**

- **Structure match**:
  - PASS: Introduction at ~2.5 pages, Literature at ~1.5 pages, Notation section present, Conclusion at ~0.5 pages
  - PASS: Contributions woven into prose rather than numbered
  - PASS: Paper organization paragraph at end of Introduction
  - DEVIATION: The venue template separates "Model Architecture" (Section 5, ~2 pages) from "Domain-Specific Methodology" (Section 6, ~2 pages). The outline merges both into a single Section 5 "Methodology" (~2.5 pages). This compresses the GAT architecture equations and the SDD debate protocol into one section, making it hard for a GNN-literate referee to find the architecture quickly and hard for a forecasting-focused referee to evaluate the debate protocol independently.
  - DEVIATION: Section 4 "Motivation for the GAT Architecture" exists and matches the template, but at ~1 page it spends most of its space on why naive combination fails (which is an empirical result, not a motivation) rather than showing data patterns that motivate the specific architectural choices (learned sparsity, regime gating, dual-head output). The SpotV2Net Section 4 uses figures of data properties to justify design decisions.
  - DEVIATION: No architecture diagram is referenced in the methodology section. The template shows a graph topology illustration in the architecture section. Figure 1 is a herding heatmap placed in Section 4, but no Figure shows the two-stage architecture itself.
  - DEVIATION: Section 6 "Empirical Results" subsections 6.1-6.7 total seven subsections, which is high for IJF. SpotV2Net uses 7.1-7.4. Consider consolidating 6.5 (Learned Graph Analysis), 6.6 (Agent Attribution), and 6.7 (Herding Dynamics) into a single interpretability section, or moving herding dynamics into the Discussion.

- **Citation style**:
  - Mostly parenthetical (Author, Year), consistent with IJF norms
  - Some references lack full bibliographic detail: "Wang et al. (2023 IJF)" should be "(Wang et al., 2023)" with the journal in the bibliography, not inline
  - "Ashery et al., 2024; Wang et al., 2025 Catfish paper" uses a nickname ("Catfish paper") that must be removed
  - "Dai et al. (2026)" appears to be a future publication; verify this is accepted/in press

---

## Project Standards Compliance

| Rule | Status | Location |
|------|--------|----------|
| No unnecessary brackets/quotes/arrows | PASS | Clean throughout |
| Minimize dash usage | MINOR VIOLATION | Lines 19, 101-103, 232: dashes used for parenthetical insertions in several places, but not excessive |
| Do not use "comprehensive" | PASS | Not found |
| Framework name = SDD | PASS | Used consistently |
| Model name = Learned Sparse GAT | VIOLATION | Title uses "Causal Graph Attention" despite project standards stating model is "Learned Sparse GAT (v3)" and the narrative strategy saying to "reframe: learned graph has causal interpretation, not Granger prior." The title as written implies Granger causality drives the graph, which is the opposite of the paper's finding. |
| Keep "Causal" in title | AMBIGUOUS | Project standards say "Keep 'Causal' in title" but also "reframe." The current title does not reframe at all; it states "Causal Graph Attention" as if the graph is causally specified, when in fact the paper's result is that causal priors fail and the graph is learned. This creates a reviewability problem: an IJF referee will expect a causal identification strategy and will be disappointed. |
| LLM debate positioned as base forecast generation, not main contribution | PASS | Introduction correctly frames GAT as the methodological contribution |
| GAT meta-aggregation as core contribution | PASS | Clear throughout |
| Failed experiments to Appendix | PASS | Appendix B covers L1, Granger prior, and MoE failures |
| Do not modify research plan without approval | PASS | Outline does not modify the research plan |

---

## Major Issues (must fix)

1. **[venue_conformity, project_standards]** at Title: The title "Causal Graph Attention for Learned Forecast Combination" is misleading. The paper's central finding regarding graph structure is that causal priors (Granger) produce dense, uninformative graphs and that the data-driven learned graph outperforms them. An IJF referee reading "Causal" in the title will expect a causal identification strategy (instrumental variables, do-calculus, or at minimum Granger with proper controls). When they discover in Appendix B that Granger causality failed, the paper's credibility suffers. The project standards note this tension but the Round 1 revision has not resolved it. Suggested fix: Replace "Causal" with "Learned Sparse" or rewrite as "Learned Graph Attention for Forecast Combination: Structured Delphi Debate and Sparse Meta-Aggregation in Oil Volatility Forecasting." If the user insists on keeping "Causal," the subtitle must clearly signal that the graph is learned, not pre-specified.

2. **[venue_conformity]** at Section 5: The methodology section merges debate protocol (domain-specific methodology) and GAT architecture (model architecture) into one section. The SpotV2Net template separates these into distinct sections because they serve different readers and different review criteria. Suggested fix: Split Section 5 into two sections. Section 5: "Learned Sparse GAT Architecture" (graph construction, multi-head GAT, regime-gated output, training protocol), roughly 2 pages. Section 6: "Structured Delphi Debate Protocol" (agent design, two-round protocol, behavioural classification, attribution), roughly 1.5-2 pages. This also places the methodological contribution (GAT) before the application-specific design (SDD), reinforcing the framing.

3. **[evidence_support]** at Section 6.2: The claim "All machine learning and deep learning baselines fail to significantly outperform HAR under HAC correction" is stated without reporting any of their DM statistics. For an IJF submission, every baseline in Table 1 needs at least one sentence interpreting its performance, or the table must be self-explanatory with significance markers. Currently the ML/DL baselines are listed in Section 6.1 and then dismissed in a single sentence in 6.2. Suggested fix: Either add a paragraph reporting the best-performing ML/DL model's DM_HAC statistic (to show the magnitude of the failure), or explicitly state that Table 1 reports all statistics and the text will focus on the key comparisons.

4. **[logical_flow]** at Section 5.2.3: The context vector description (9-dimensional) is placed after the final prediction equation, creating a backward reference. A reader encountering the regime gate's three context features on first reading does not yet know where they come from, and then finds a 9-dimensional context vector described afterwards with no explanation of how it feeds into the architecture. Suggested fix: Introduce the context vector at the beginning of Section 5.2 alongside the node features, before describing any component that uses context features.

5. **[venue_conformity]** at Figures plan: No architecture diagram is planned for the methodology section. An architecture diagram (two-stage pipeline, showing data flow from raw features through SDD agents through the GAT to the final prediction) is standard for any GNN paper in IJF and would be the most referenced figure in the paper. The current Figure 1 is a herding heatmap, which is a supporting figure, not the primary method illustration. Suggested fix: Add a new Figure 1 showing the two-stage architecture (demoting the herding heatmap to Figure 2 or later). The venue template explicitly includes "Graph topology illustration" as a standard figure.

---

## Minor Issues (should fix)

1. **[writing_quality]** at Section 2.5, lines 239-244: The herding definition and its 42.9% statistic are empirical results from the author's own data, placed in the Literature Review. The literature review should establish that herding is a known phenomenon (with citations), not report own-data statistics. Suggested fix: Move the "42.9% of agent-day observations" finding and the "technical and cross_market agents herding most frequently (52%)" statistics to Section 6.7, and keep Section 2.5 focused on the published literature on emergent bias.

2. **[writing_quality]** at Section 4.1: "The confidence-weighted mean (naive aggregation) achieves RMSE = 0.1512 after HAC-corrected evaluation, which is not significantly better than HAR (RMSE = 0.1549, DM_HAC = -1.494, p = 0.135)." Reporting numeric coefficients in the motivation section before the reader has seen the experimental setup is premature. The motivation section should use qualitative arguments and data patterns, not results. Suggested fix: Rephrase as "Preliminary analysis shows that the confidence-weighted mean does not significantly outperform HAR after correcting for overlapping-horizon autocorrelation (results in Section 6)." Reserve the exact numbers for the results section.

3. **[writing_quality]** at multiple locations: The pattern "Include Figure X here: [description]" appears four times. While acceptable in an outline, if this is intended as near-final prose, these must become proper figure references. This is a minor formatting issue given the outline stage.

4. **[argument_clarity]** at Section 1 "Contributions": The third contribution (HAC correction matters) is an empirical finding about evaluation methodology, not a methodological contribution of this paper. HAC-corrected DM tests are well established (Newey and West, 1987; Harvey et al., 1997). The contribution is in demonstrating that it changes the ranking in this specific context. Suggested fix: Reframe the third contribution as "demonstrating that proper treatment of overlapping-horizon autocorrelation reverses the apparent significance of naive LLM debate forecasts, establishing that the learned combination mechanism, not the base forecast quality, is the source of genuine forecasting gains."

5. **[citation_coverage]** at Section 2.1: The forecast combination literature lacks several important recent references. Petropoulos et al. (2022, IJF) on forecasting in practice is a natural anchor. The "forecast combination puzzle" is attributed to Genre et al. (2013) and Claeskens et al. (2016), but the canonical treatment is Smith and Wallis (2009, Oxford Bulletin). Suggested fix: Add Petropoulos et al. (2022) and Smith and Wallis (2009) to the combination section.

6. **[citation_coverage]** at Section 2.3: Abolghasemi et al. (2025 IJF) is the only IJF citation in the LLM forecasting subsection. Given that this is an IJF submission, anchoring more heavily in the journal's own literature would be strategic. Suggested fix: Check whether Makridakis et al. (M-competitions) or other IJF-published LLM/ML forecasting evaluations exist and cite them.

7. **[logical_flow]** at Section 6.1: Baselines are listed as "five categories" but then enumerate six lines (Naive, Econometric, ML, DL, LLM-only, and the proposed method). The proposed method is not a baseline. Suggested fix: Say "five baseline categories" and list the proposed method separately.

8. **[statistical_rigor]** at Section 3.2: The walk-forward protocol says "The walk-forward evaluation produces 972 out-of-sample predictions" but this appears in the Problem Formulation section, not in the Empirical Results. This is fine, but note that n = 972 is the walk-forward sample while n = 1,224 is the clean sample. The abstract says "Evaluated on 1,224 trading days" which could be misread as all 1,224 having walk-forward predictions. Suggested fix: Clarify in the abstract that 1,224 is the clean sample and 972 is the walk-forward evaluation sample, or simply report 972 in the abstract since that is the number of out-of-sample predictions.

9. **[writing_quality]** throughout: Scholar-as-subject constructions occur frequently. Section 2.1 alone has "Wang et al. (2023 IJF) provide," "Bates and Granger (1969)," and subject-position citation twice in three sentences. The venue template allows a maximum of approximately two per paragraph. Suggested fix: Convert most to parenthetical form. For example, "A recent structured review identifies three mechanisms..." instead of "Wang et al. (2023 IJF) provide the most recent structured review, identifying three mechanisms..."

10. **[project_standards]** at line 89: "Wang et al., 2025 Catfish paper" uses a nickname/label. This violates the "no unnecessary brackets, quotes, arrows" standard broadly interpreted, and is unprofessional for a journal submission. Suggested fix: Use the actual paper title or simply "(Wang et al., 2025)."

11. **[evidence_support]** at Section 6.5: "Multiple locally optimal sparse graphs exist, all achieving similar out-of-sample performance." This is a strong claim. It would be strengthened by reporting the RMSE variance across seeds, which the project data shows is available (seed variance = 0.0040 for v3). Suggested fix: Add "The 5-seed ensemble exhibits RMSE standard deviation of 0.004 across seeds despite topologically distinct graphs, supporting the claim of multiple equivalent optima."

12. **[argument_clarity]** at Section 5.1.2: "The SDD protocol maps expert human analytical practice onto a reproducible LLM inference procedure" is a strong claim that is not supported by any evidence in the outline. No human expert comparison, no domain expert validation, and no citation establishing that oil analysts actually follow a two-round structured debate. Suggested fix: Either provide a citation for structured elicitation in oil markets or soften to "The SDD protocol adapts the Delphi elicitation method to a reproducible multi-agent LLM inference procedure."

13. **[venue_conformity]** at Section 7 "Discussion": The Discussion runs ~1.5 pages with four subsections, which is slightly light for IJF. The SpotV2Net paper embeds limitations within the Discussion and includes a "practical implications" subsection. Section 7.3 on interpretability could be expanded with a concrete worked example showing how a risk manager would use the Shapley output on a specific date. Suggested fix: Expand Section 7.3 with one concrete example, and merge Section 7.4 Limitations into 7.1-7.3 as closing paragraphs rather than a standalone section.

14. **[writing_quality]** at Section 1 "The LLM opportunity and the aggregation bottleneck": The final sentence "This is the aggregation bottleneck: the value is in the generation, but naive combination squanders it" is a good framing sentence but uses a colon followed by a clause that reads as a slogan. IJF prose tends to be more measured. Suggested fix: "This aggregation bottleneck means that the forecasting value created by diverse LLM agents is largely dissipated by naive combination."

---

## Strengths

1. The reframing from "Causal-GAT for interpretable forecasting" (original) to "learned forecast combination via graph structure" (Round 1) is the correct strategic move for IJF. The forecast combination literature anchor (Wang et al., 2023) gives the paper a natural home in the journal's scope.

2. The honest treatment of statistical significance is a major asset. Reporting that naive debate is NOT significant after HAC correction, and that only the learned GAT survives, is exactly the kind of self-critical analysis that IJF referees value. Most submitted papers would hide the HAC correction.

3. The motivation section (Section 4) is a genuine improvement over the original. Explaining why naive combination fails before presenting the solution gives the reader a problem to solve, not just a method to absorb.

4. The Appendix B on failed optimisation experiments is excellent journal practice. Documenting that L1, Granger priors, and MoE all failed saves future researchers from repeating dead ends and demonstrates thoroughness.

5. The GAT vs MLP ablation (DM_HAC = -2.825, p = 0.005) is the single strongest piece of evidence in the paper. It isolates the graph structure contribution from confounds, and the outline correctly places it in a dedicated subsection (6.3).

6. All key numbers in the outline are consistent with the project data provided for cross-checking. No numerical discrepancies were found.

---

## Verdict: NEEDS_REVISION

The outline scores 3.4/5.0, below the 4.0 convergence threshold. The title-content mismatch (Major Issue 1) and the methodology section structure (Major Issue 2) are the most consequential problems. The title issue risks a desk rejection if a handling editor interprets "Causal Graph Attention" as a causal inference paper and assigns causal-methods referees. The methodology structure issue makes the paper harder to evaluate for both the GNN-specialist and the forecasting-specialist referee. Both are fixable in a Round 2 revision.

No veto conditions are triggered (venue_conformity = 3.0 > 2.0, project_standards = 3.0 > 2.0), but the paper is not yet at submission quality.
