# Project Writing Standards

Extracted from CLAUDE.md and project context. Updated 2026-03-26.

## Hard Constraints

1. **No unnecessary brackets, quotes, arrows** - avoid AI-typical punctuation patterns
2. **Minimize dash usage** for punctuation
3. **Do not use "comprehensive"** - banned word
4. **Framework name**: Structured Delphi Debate (SDD)
5. **Model name**: DropEdge GAT (primary), Learned GAT (secondary/robustness)
6. **Prediction target**: Forward-looking 20-day realized volatility (fwd_rv_20d)
7. **Do not modify research plan without user approval**

## Narrative Strategy (confirmed by user)

1. **"Causally interpretable" framing** - learned graph has causal interpretation, NOT Granger prior
2. **Failed experiments go to Appendix** - MoE collapse, L1 failure, causal prior comparison
3. **IJF framing**: Core contribution is "learned graph-based forecast combination method"
4. **LLM debate** positioned as "how we generate diverse base forecasts", not the main contribution
5. **GAT meta-aggregation** is the methodological contribution, connecting to forecast combination literature (Wang et al. 2023)

## Key Numbers (embargo-corrected, aligned sample n=972)

### Full model comparison (Table 1)
- DropEdge GAT: RMSE=0.1171, DM_HAC=-3.212 (p=0.0013) ***
- Learned GAT: RMSE=0.1188, DM_HAC=-3.165 (p=0.0016) **
- GARCH(1,1): RMSE=0.1366, DM_HAC=-2.604 (p=0.009) **
- MLP no-graph: RMSE=0.1444, DM_HAC=-1.214 (p=0.225) n.s.
- Ridge: RMSE=0.1493, DM_HAC=-0.545 (p=0.586) n.s.
- Single agent: RMSE=0.1495, DM_HAC=-1.230 (p=0.219) n.s.
- Debate (naive): RMSE=0.1512, DM_HAC=-1.494 (p=0.135) n.s.
- HAR: RMSE=0.1549 (baseline)
- Persistence: RMSE=0.1551, DM_HAC=0.192 (p=0.848) n.s.
- RF: RMSE=0.1605, GBR: RMSE=0.1645, Lasso: RMSE=0.1731
- XGBoost: RMSE=0.1809, DM_HAC=1.463 (p=0.144) n.s.
- HistMean: RMSE=0.1957
- Transformer: RMSE=0.1985, DM_HAC=1.691 (p=0.091) n.s.
- LSTM: RMSE=0.2743, DM_HAC=3.446 (p=0.001) *** (worse than HAR)

### Pairwise DM
- GAT vs MLP: DM_HAC=-3.336 (p=0.0009)
- GAT vs GARCH: DM_HAC=-1.948 (p=0.051) marginally significant
- GAT vs Dense GAT: DM_HAC=-6.382 (p<0.001)
- GAT vs Identity: DM_HAC=-1.107 (p=0.268) not significant
- Dense vs Identity: DM_HAC=6.365 (p<0.001) Identity much better

### Ablation results (Table 2, ablation_results_v2.csv)
- DropEdge GAT (full): RMSE=0.1171, DM=-3.212 (p=0.001)
- Identity (self-loops only): RMSE=0.1188, DM=-3.165 (p=0.002)
- No regime gate: RMSE=0.1198, DM=-2.995 (p=0.003)
- Random graph (~16/42): RMSE=0.1414, DM=-1.384 (p=0.166) n.s.
- Dense GAT (42/42): RMSE=0.1452, DM=-1.185 (p=0.236) n.s.
- No Shapley/Myerson: RMSE=0.1520, DM=-0.420 (p=0.675) n.s.
- "No behaviour" DELETED due to Shapley look-ahead leakage

### Regime breakdown (Table 3)
| Regime | GAT | GARCH | MLP | HAR | Debate | n |
|--------|-----|-------|-----|-----|--------|---|
| Low | 0.0789 | 0.1430 | 0.1280 | 0.2004 | 0.1950 | 65 |
| Normal | 0.1243 | 0.1456 | 0.1543 | 0.1479 | 0.1406 | 454 |
| Elevated | 0.1083 | 0.1223 | 0.1401 | 0.1362 | 0.1327 | 377 |
| High | 0.1380 | 0.1423 | 0.1177 | 0.2232 | 0.2334 | 76 |

### Baseline classification
- Naive: persistence
- Econometric: HAR, GARCH(1,1)
- ML: Ridge, Lasso, RF, GBR, XGBoost
- DL: LSTM, Transformer
- LLM: single-agent, naive debate
- Proposed: DropEdge sparse GAT

### Regime definition
Fixed thresholds on persist_vol: low (<0.20), normal (0.20-0.35), elevated (0.35-0.55), high (>0.55)
NOT percentile cutoffs.

### Edge analysis
- 15 walk-forward windows x 5 seeds = 75 trained models (not 80, first window skipped by embargo)
- ~16/42 active edges per model
- No edge >50% across all seeds/windows
- Top temporal shifts: technical→geopolitical -47pp, sentiment→macro_demand +40pp, sentiment→supply_opec -40pp

### Group behavior (aligned sample n=972)
- Herding: 43.4%, technical/cross_market highest (52%)
- All 7 agents weakly negative mean Shapley (helpful), positive on 38-59% of days
- Communication density vs herding: r=0.35
- Partial r after regime control: <0.08 (not significant)

## Statistical Rigor Requirements

- Must use Newey-West HAC DM test (Bartlett kernel, bandwidth=19)
- h=20 overlapping windows cause high autocorrelation (ACF lag1=0.938)
- Naive DM overestimates significance by ~65%
- _error columns store squared errors, not raw errors
- Only DropEdge GAT, Learned GAT, and GARCH significantly outperform HAR
