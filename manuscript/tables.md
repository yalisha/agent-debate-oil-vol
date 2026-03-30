# Tables

## Table 1: Full Model Comparison

RMSE and HAC-corrected Diebold-Mariano statistics for all 16 models on the aligned evaluation sample ($n = 972$). The DM test uses Newey-West HAC standard errors with Bartlett kernel and bandwidth 19. Significance: \* $p < 0.05$, \*\* $p < 0.01$, \*\*\* $p < 0.001$.

| Category | Model | RMSE | $\Delta$% vs HAR | DM$_{\text{HAC}}$ | $p$ | Sig |
|----------|-------|:----:|:---------:|:--------:|:---:|:---:|
| Proposed | DropEdge GAT | 0.1171 | $-$24.4 | $-$3.212 | 0.001 | \*\*\* |
| Proposed | Learned GAT | 0.1188 | $-$23.3 | $-$3.165 | 0.002 | \*\* |
| Econometric | GARCH(1,1) | 0.1366 | $-$11.8 | $-$2.604 | 0.009 | \*\* |
| Proposed | MLP no-graph | 0.1444 | $-$6.8 | $-$1.214 | 0.225 | |
| ML | Ridge | 0.1493 | $-$3.6 | $-$0.545 | 0.586 | |
| LLM | Single agent | 0.1495 | $-$3.5 | $-$1.230 | 0.219 | |
| LLM | Debate (naive) | 0.1512 | $-$2.4 | $-$1.494 | 0.135 | |
| Econometric | HAR | 0.1549 | --- | --- | --- | baseline |
| Naive | Persistence | 0.1551 | +0.1 | 0.192 | 0.848 | |
| ML | RF | 0.1605 | +3.6 | | | |
| ML | GBR | 0.1645 | +6.2 | | | |
| ML | Lasso | 0.1731 | +11.7 | | | |
| ML | XGBoost | 0.1809 | +16.8 | 1.463 | 0.144 | |
| Naive | HistMean | 0.1957 | +26.3 | | | |
| DL | Transformer | 0.1985 | +28.1 | 1.691 | 0.091 | |
| DL | LSTM | 0.2743 | +77.1 | 3.446 | 0.001 | \*\*\* $\dagger$ |

$\dagger$ Significantly *worse* than HAR.

---

## Table 2: Architecture Ablation Results

RMSE and HAC-corrected DM statistics for architectural variants on the aligned evaluation sample ($n = 972$). All variants share the walk-forward protocol, 5-seed ensemble, 20-day label embargo, and HAC inference of Section 3.2.

| Variant | Ablation target | RMSE | DM$_{\text{HAC}}$ | $p$ | Sig |
|---------|----------------|:----:|:--------:|:---:|:---:|
| DropEdge GAT (full) | --- | 0.1171 | $-$3.212 | 0.001 | \*\*\* |
| Identity (self-loops) | Inter-agent edges | 0.1188 | $-$3.165 | 0.002 | \*\* |
| No regime gate | Regime-gated dual head | 0.1198 | $-$2.995 | 0.003 | \*\* |
| Random graph (~16/42) | Learned topology | 0.1414 | $-$1.384 | 0.166 | |
| Dense GAT (42/42) | Learned sparsity | 0.1452 | $-$1.185 | 0.236 | |
| No Shapley/Myerson | Attribution features | 0.1520 | $-$0.420 | 0.675 | |
| MLP no-graph | Graph structure | 0.1444 | $-$1.214 | 0.225 | |

---

## Table 3: Regime-Conditional RMSE

RMSE by volatility regime for the DropEdge GAT and four representative comparators. Regimes are defined by fixed thresholds on persistence volatility: low ($< 0.20$), normal ($0.20$--$0.35$), elevated ($0.35$--$0.55$), high ($> 0.55$).

| Regime | $n$ | DropEdge GAT | GARCH | MLP | HAR | Debate |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Low | 65 | **0.0789** | 0.1430 | 0.1280 | 0.2004 | 0.1950 |
| Normal | 454 | **0.1243** | 0.1456 | 0.1543 | 0.1479 | 0.1406 |
| Elevated | 377 | **0.1083** | 0.1223 | 0.1401 | 0.1362 | 0.1327 |
| High | 76 | 0.1380 | 0.1423 | **0.1177** | 0.2232 | 0.2334 |
| **All** | **972** | **0.1171** | 0.1366 | 0.1444 | 0.1549 | 0.1512 |

Bold indicates lowest RMSE within each regime.
