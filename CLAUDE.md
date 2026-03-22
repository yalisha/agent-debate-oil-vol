# Multi-Agent LLM Debate for Oil Volatility Forecasting

## Project Overview
研究多Agent LLM辩论系统中的群体行为偏差，以原油波动率预测为实证场景。框架名称: Structured Delphi Debate (SDD)。

## Key Architecture

### Prediction Target
前瞻性20日已实现波动率 (forward-looking realized volatility):
`fwd_rv_20d = std(log_returns[t+1:t+21]) * sqrt(252)`

### Core Formula
`pred = persistence_vol + weighted_mean(agent_adjustments)`
其中 persistence_vol = 当前20日已实现波动率，agent输出adjustment而非绝对值。

### 7 Specialist Agents
geopolitical, macro_demand, monetary, supply_opec, technical, sentiment, cross_market

### Debate Protocol
2轮: Round 1独立预测 -> Round 2看到其他agent后修正 -> confidence-weighted数学聚合

## Data

### Sources
- `data/oil_macro_daily.csv`: WTI价格、VIX、DXY、利率、波动率指标 (2020-01 ~ 2025-06)
- `data/gdelt_daily_features.csv`: GDELT地缘政治事件特征
- `data/新gold_prediction_features.csv`: 原始数据源

### Results (Current)
- `results/debate_eval_full_20260320_2343.csv`: Gemini 3 Flash前瞻性h=20全样本 (n=1285)
- `results/debate_attribution_full_20260320_2343.json`: 对应Shapley归因
- `results/vol_baselines_full.csv`: ML baselines (Ridge/Lasso/GBR/RF) **旧target, 需重跑**
- `results/vol_baselines_dl_rolling.csv`: DL baselines (XGBoost/LSTM/Transformer) **旧target, 需重跑**

### Known Data Issues
- 2020-04-20 WTI负油价 (-$36.98) 污染前后各约20天的波动率计算
- 分析时需排除 actual_vol > 2 或 persist_vol > 2 的61天
- `_error` 列存储的是**平方误差** `(pred - actual)^2`，不是原始误差

## Key Scripts

### Core System
- `src/debate_system.py`: DebateEngine, SingleAgentBaseline, AttributionEngine
  - API: Gemini 3 Flash via `https://api.akane.win/v1`
  - Model: `gemini-3-flash-preview`
  - max_tokens=2000 (Gemini thinking tokens需要足够空间)
- `src/run_debate_eval.py`: 全样本评估脚本，支持 `--mode full/test`, `--horizon 5/20`, `--resume`
  - 3天并行处理 (ThreadPoolExecutor)
  - 每个线程独立创建engine实例，write_lock保护checkpoint

### Data Preparation
- `src/prepare_oil_data.py`: 从原始CSV提取WTI价格序列，计算收益率、波动率、前瞻性目标

### Analysis
- `src/deep_analysis.py`: 6模块深化分析 (case study, regression, intervention, Shapley, cascade, comparison)
  - 输出: `docs/` 下9个PDF + 8个LaTeX表
- `src/run_vol_baselines.py`: ML baselines (Ridge/Lasso/GBR/RF)
- `src/run_vol_baselines_dl_rolling.py`: DL baselines (XGBoost/LSTM/Transformer)

### Claude Code Sub-Agent Demo
- `claude/run_claude_debate.py`: 用Claude Code子agent执行debate的helper脚本
- `claude/feb2022_*.json`: 2022年2月demo数据和结果

## Running Evaluations

```bash
# Full evaluation (forward-looking h=20)
cd /Users/mac/computerscience/17Agent可解释预测
/opt/miniconda3/bin/python -u src/run_debate_eval.py --mode full --horizon 20 --rounds 2 --attrib-samples 500 --resume

# Deep analysis (after evaluation completes)
/opt/miniconda3/bin/python src/deep_analysis.py

# ML baselines (needs update for forward-looking target)
/opt/miniconda3/bin/python src/run_vol_baselines.py
```

## Current Status (2026-03-22)
- Gemini 3 Flash forward-looking h=20 全样本完成 (n=1285, 0 failures)
- Deep analysis已重跑，排除负油价污染61天
- ML/DL baselines用的是旧shift(-1) target，需用新forward-looking target重跑
- 方法论框架已完成文献锚定 (docs/methodology_framework.md)

## Key Results
- Debate RMSE: 0.1873 (n=1224, 排除负油价)
- Debate vs Persistence: DM=-2.44, p=0.015 (显著)
- Debate vs Single: DM=-0.11, p=0.913 (不显著)
- Herding占比: 43%, 危机期升至47-48%
- 所有7个agent Shapley值均为负 (均改善预测)

## Documentation
- `docs/research_framework.md`: 研究框架 (研究问题、理论、方法、结果)
- `docs/methodology_framework.md`: 文献锚定的9步方法论
- `docs/prediction_mechanism.md`: 预测机制详解
- `bib/references.bib`: 参考文献库
- `outline.md`: 论文大纲
