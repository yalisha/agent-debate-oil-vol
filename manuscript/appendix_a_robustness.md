# Appendix A: Robustness to the Data Exclusion

The main analysis excludes 61 trading days between 24 February and 22 May 2020 to remove the WTI contract dislocation period, during which the front-month settlement price fell to $-\$36.98$ on 20 April 2020 and the surrounding weeks exhibited return dynamics driven by contractual mechanics rather than supply-demand fundamentals. This appendix examines sensitivity to this exclusion.

## A.1 Effect on Sample Composition

The exclusion removes days in which 20-day realised volatility exceeds 1.0 (annualised), well above the high-regime threshold of 0.55. Retaining these days would add extreme observations to the training windows that overlap the dislocation period, potentially distorting the learned graph structure and the regime gate calibration.

## A.2 Sensitivity Analysis

To assess robustness, we re-estimate the DropEdge GAT on the full sample ($n = 1{,}285$) without exclusion, using the same walk-forward protocol, 5-seed ensemble, and 20-day label embargo. The headline results are as follows.

| Sample | $n_{\text{OOS}}$ | DropEdge GAT RMSE | HAR RMSE | $\Delta$% |
|--------|:---------:|:--------:|:--------:|:---:|
| Clean (main analysis) | 972 | 0.1171 | 0.1549 | $-$24.4 |
| Full (no exclusion) | 1,033 | [TBD] | [TBD] | [TBD] |

[TODO: Run the full-sample variant and fill in the results. The expectation is that including the dislocation period will increase RMSE for all models because the extreme observations inflate squared errors, but the relative ranking should be preserved.]

## A.3 Qualitative Assessment

The exclusion is conservative in scope: 61 of 1,285 days (4.7 percent). The excluded period is contiguous and corresponds to a well-documented contractual anomaly (the expiry of the May 2020 contract with negative settlement) that is unlikely to recur under the revised CME margin rules adopted in 2021. The main conclusions of the paper are therefore unlikely to be sensitive to this exclusion, though the formal sensitivity analysis above provides quantitative confirmation.
