# IJF Venue Template

Extracted from: SpotV2Net (Brini & Toscano, IJF 2025, Vol. 41, pp. 1093-1111)
Cross-referenced with: Zhang et al. (IJF 2025), Wang et al. (IJF 2023)

## Structural Template

- **Total length**: 19 pages (journal two-column format), approximately 8000-9000 words
- **Abstract**: ~150 words, structured as: problem -> method -> empirical scope -> key result -> interpretability contribution
- **Keywords**: 5-6 terms
- **Dataset link**: Required (GitHub or similar)

## Section Structure (from SpotV2Net)

1. **Introduction** (~2.5 pages)
   - Practical motivation (why this prediction matters)
   - Existing approaches and their limitations
   - Brief description of proposed method
   - Contributions (implicitly woven into text, not numbered list)
   - Paper organization paragraph at end

2. **Literature Review** (~1.5 pages, Section 2)
   - Organized by topic stream, not chronologically
   - Focused and concise, only directly relevant work
   - IJF editorial: "Do not include references simply to increase the length of your bibliography"

3. **Notation and Assumptions** (~1 page, Section 3)
   - Formal mathematical setup before methodology
   - Clear variable definitions

4. **Motivation for Architecture** (~1 page, Section 4)
   - Why this specific architecture (GAT) is appropriate
   - Intuitive argument before formal details
   - Figures showing data patterns that motivate the approach

5. **Model Architecture** (~2 pages, Section 5)
   - Full equations for GAT layers, attention mechanism, multi-head attention
   - Edge features incorporation
   - Architecture diagram (Figure)

6. **Domain-Specific Methodology** (~2 pages, Section 6)
   - Fourier estimators in SpotV2Net; corresponds to debate protocol + attribution in our paper

7. **Empirical Application** (~5 pages, Section 7)
   - 7.1 Data preparation and estimation
   - 7.2 Single-step forecast results (Tables with DM test matrices)
   - 7.3 Multi-step forecast results
   - 7.4 Interpretability analysis (GNNExplainer subgraphs)

8. **Conclusion** (~0.5 page, Section 8)
   - Summary of method and findings
   - Limitations mentioned briefly
   - Future directions

## Tables and Figures Pattern

- **Tables**:
  - Data split summary (train/val/test periods)
  - Aggregate metrics (MSE, QLIKE) for all models
  - DM test pairwise comparison matrices
  - Hyperparameter settings
- **Figures**:
  - Time series visualization of key features
  - Graph topology illustration (3-node example)
  - GNNExplainer subgraph visualizations
  - Multi-step forecast comparison plots

## Citation Style
- Parenthetical: (Author, Year) dominant
- Author-as-subject used sparingly: "Veličković et al. (2017) proposed..."
- Max ~2 author-as-subject sentences per paragraph

## IJF-Specific Editorial Requirements
- Double-blind review
- Data + code must be provided (reproducibility check before acceptance)
- "Simply applying a given modeling and forecasting approach to a given dataset is not in itself sufficient"
- Must demonstrate "new knowledge and insight in the science and practice of forecasting"
- Simple notation preferred: "our readership crosses many disciplines"
- Concise writing: "Referees will be asked to consider the value of the paper relative to its length"
- Only relevant references
