# Embargo Revalidation Report (2026-03-24)

## Purpose

This note documents the protocol bug we found in the forward-looking target evaluation, the code changes applied, the scripts re-run under the corrected protocol, and the updated benchmark ranking.

Prepared for follow-up discussion and model-selection decisions.

## Executive Summary

There **was** a serious evaluation issue.

The project predicts a forward-looking 20-day realized volatility target:

`y_t = RV(t+1 : t+20)`

For this target, the label for sample `t` is only fully observed at time `t+20`.  
Therefore, when forecasting day `t`, any supervised model must **not** train on labels newer than `t-20`.

Before the fix, multiple supervised scripts used training data up to `t-1` or up to the current retrain boundary. That creates label leakage for all supervised models using `fwd_rv_20d`, including simple ML baselines and learned meta-aggregation models.

After adding a unified `20-day label embargo` and re-running key scripts:

- The previously strong simple ML baselines dropped sharply.
- The corrected `Learned GAT` remained strong.
- The corrected best model is now `DropEdge GAT`.
- Post-hoc stacking is no longer the best corrected result.

## What Was Fixed

### Shared fix

Added a shared helper:

- `src/walk_forward_utils.py`

Core rule:

- `LABEL_EMBARGO = 20`
- when forecasting split index `t`, supervised training ends at `t - 20 + 1` (exclusive index form)

### Patched scripts

The corrected embargo logic was applied to:

- `src/run_vol_baselines.py`
- `src/run_vol_baselines_dl_rolling.py`
- `src/causal_gat_aggregation.py`
- `src/run_ablations.py`
- `src/hybrid_sparse_gat.py`
- `src/optimized_gat.py`
- `src/eval_protocol.py`
- `src/final_comparison.py`
- `src/further_optimization.py`
- `src/final_model.py`
- `src/moe_meta_aggregation.py`

## Re-run Scope Completed

The following scripts were re-run under the corrected protocol:

- `src/run_vol_baselines.py`
- `src/run_vol_baselines_dl_rolling.py`
- `src/final_comparison.py`
- `src/optimized_gat.py`
- `src/further_optimization.py`

## Corrected Results

### 1. Corrected Learned GAT vs No-Graph Baseline

Source:

- `results/final_comparison.csv`

| Model | Ensemble RMSE | DM_HAC vs HAR | p-value | Per-seed Mean | Per-seed Std |
|---|---:|---:|---:|---:|---:|
| Learned GAT | **0.1188** | **-3.165** | **0.0016** | 0.1266 | 0.0039 |
| MLP no-graph | 0.1444 | -1.214 | 0.2246 | 0.1530 | 0.0013 |

Direct comparison:

- `GAT vs MLP`: `DM_HAC = -3.336`, `p = 0.0009`

Interpretation:

- Under the corrected protocol, graph structure still adds clear value.
- The no-graph MLP is no longer competitive with the learned GAT.

### 2. Corrected ML Baselines Aligned to GAT Dates

These are re-evaluated on the same `972` dates used in the corrected GAT comparison.

| Model | n | RMSE | DM_HAC vs HAR | p-value |
|---|---:|---:|---:|---:|
| Ridge | 972 | 0.1493 | -0.548 | 0.5836 |
| RF | 972 | 0.1605 | 0.437 | 0.6623 |
| GBR | 972 | 0.1645 | 0.792 | 0.4284 |
| Lasso | 972 | 0.1731 | 1.118 | 0.2637 |
| HistMean | 972 | 0.1957 | 0.928 | 0.3537 |

Interpretation:

- The earlier “simple ML beats GAT” impression was mainly a protocol artifact.
- After embargo correction, **no simple ML baseline beats corrected GAT**.
- None of the aligned ML baselines is statistically significant vs HAR under HAC correction.

### 3. Corrected DL Baselines Aligned to GAT Dates

These are also aligned to the same `972` GAT dates.

| Model | n | RMSE | DM_HAC vs HAR | p-value |
|---|---:|---:|---:|---:|
| XGBoost | 972 | 0.1809 | 1.462 | 0.1438 |
| Transformer | 972 | 0.1985 | 1.693 | 0.0905 |
| LSTM | 972 | 0.2743 | 3.450 | 0.0006 |

Interpretation:

- Corrected DL baselines are all weaker than corrected GAT.
- The LSTM is significantly worse than HAR.

### 4. Corrected Optimized GAT Sweep

Source:

- `results/optimized_gat_results.csv`

Configuration summary:

| Config | Per-seed RMSE Mean | Per-seed RMSE Std | Ensemble RMSE | Mean DM_HAC | Mean Active Edges |
|---|---:|---:|---:|---:|---:|
| Learned-H16 | **0.1266** | 0.0044 | **0.1188** | -2.541 | 16 |
| SparsePrior-H16 | 0.1279 | 0.0038 | 0.1208 | -2.324 | 14 |
| DensePrior-H16 | 0.1279 | 0.0038 | 0.1208 | -2.324 | 41 |

Interpretation:

- The corrected best config in this sweep remains `Learned-H16`.
- Learned graph still beats both sparse and dense prior warm-starts.
- Priors still do not add value under the corrected protocol.

### 5. Corrected Further Optimization Results

Source:

- `results/further_optimization_results.csv`

| Method | Ensemble RMSE | DM_HAC | p-value | low | normal | elevated | high |
|---|---:|---:|---:|---:|---:|---:|---:|
| DropEdge GAT | **0.1171** | **-3.212** | **0.0013** | 0.0789 | 0.1243 | 0.1083 | 0.1380 |
| Learned GAT | 0.1188 | -3.165 | 0.0016 | 0.0768 | 0.1285 | 0.1094 | 0.1309 |
| Enhanced Features GAT | 0.1189 | -3.172 | 0.0015 | 0.0798 | 0.1262 | 0.1126 | 0.1300 |
| Regime-Weighted GAT | 0.1225 | -3.002 | 0.0027 | 0.0816 | 0.1313 | 0.1129 | 0.1406 |
| MLP no-graph | 0.1444 | -1.214 | 0.2246 | 0.1280 | 0.1543 | 0.1401 | 0.1177 |

Additional corrected stacking result from script output:

| Method | Ensemble RMSE | DM_HAC | p-value |
|---|---:|---:|---:|
| Stacked (GAT+MLP) | 0.1227 | -3.232 | 0.0012 |

Interpretation:

- Under the corrected protocol, `DropEdge GAT` is the new best model among re-run methods.
- `Enhanced Features GAT` is nearly tied with baseline `Learned GAT`, but does not beat `DropEdge GAT`.
- `Regime-Weighted GAT` still underperforms.
- `Stacked (GAT+MLP)` is no longer the best result. It is now weaker than both `DropEdge GAT` and baseline `Learned GAT`.

## Main Conclusions

### Conclusion 1: The earlier simple-ML win was not reliable

The apparent advantage of simple ML models came from a flawed evaluation protocol for a forward-looking target.  
After introducing the required 20-day label embargo, their performance dropped materially.

### Conclusion 2: The graph-based model survives protocol correction

This is the most important result.

The corrected `Learned GAT` still performs well:

- strong RMSE
- significant HAC-DM vs HAR
- significant direct win vs no-graph MLP

So the graph result was **not** just a leakage artifact.

### Conclusion 3: The best corrected model is now DropEdge GAT

Among the re-run optimized variants:

- `DropEdge GAT` is best by RMSE: `0.1171`
- It remains significant vs HAR
- It beats the plain corrected Learned GAT

This makes `DropEdge GAT` the strongest current candidate for the main model.

### Conclusion 4: Post-hoc stacking should no longer be the headline method

Before correction, stacking looked like the strongest option.  
After correction:

- `Stacked (GAT+MLP) = 0.1227`
- `DropEdge GAT = 0.1171`

So stacking is no longer the top result. It can still be discussed as a robustness / complementarity experiment, but it should not be the main method claim.

### Conclusion 5: Learned graph still matters more than topology priors

The corrected sweep still supports the same qualitative story:

- learned graph > fixed causal prior
- sparse prior and dense prior remain effectively tied
- graph helps, but the exact hand-crafted prior graph still does not

## Recommended Model Positioning

### Recommended primary model

Use:

- `DropEdge GAT`

Why:

- best corrected RMSE among the re-run variants
- significant vs HAR
- preserves the learned-graph story

### Recommended secondary / robustness model

Keep:

- `Learned GAT (Learned-H16)`

Why:

- very close in performance
- simpler story
- already validated in corrected direct comparison vs MLP

### Recommended models to de-emphasize

- `Post-hoc stacking`: not best anymore
- `Regime-weighted loss`: consistently weaker
- `Simple ML` and `DL` baselines: no longer threaten the corrected graph result

## What Should Be Updated in Paper/Notes

Claude should update any narrative that still says:

- simple ML beats the graph model
- stacking is the best overall approach
- old RMSE numbers from pre-embargo scripts are final

Instead, the corrected narrative is:

1. A forward-label leakage issue was identified and fixed via a 20-day embargo.
2. After correction, simple ML/DL baselines are no longer dominant.
3. Graph-based learned aggregation remains strong and statistically meaningful.
4. `DropEdge GAT` is currently the best corrected model.

## Files That Are Safe to Cite Now

- `results/final_comparison.csv`
- `results/optimized_gat_results.csv`
- `results/further_optimization_results.csv`
- `results/vol_baselines_full.csv`
- `results/vol_baselines_dl_rolling.csv`

## Files That Should Be Treated Carefully

The following files may still contain **old-protocol** outputs unless explicitly re-run after the embargo fix:

- `results/ablation_results.csv`
- `results/causal_gat_results.csv`
- `results/hybrid_sparse_gat_results.csv`
- `results/final_model_results.csv`
- `results/moe_results.csv`

These should not be treated as final corrected evidence yet.

## Suggested Next Steps

1. Update `CLAUDE.md` and any draft text to reflect the corrected benchmark ordering.
2. Treat `DropEdge GAT` as the provisional main model.
3. Treat `Learned GAT` as the clean robustness reference.
4. Re-run any remaining old-protocol result files only if they are needed for the paper.
5. Avoid citing pre-embargo simple-ML superiority claims.
