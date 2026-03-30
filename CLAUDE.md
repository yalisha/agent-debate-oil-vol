# Multi-Agent LLM Debate for Oil Volatility Forecasting

## Project Overview
研究多Agent LLM辩论系统中的群体行为偏差，以原油波动率预测为实证场景。框架名称: SDD-GAT (Structured Delphi Debate + Graph Attention Network)。通过learned meta-aggregation升级聚合机制。目标期刊: International Journal of Forecasting (IJF, AJG 3星), 备选 Journal of Forecasting (JoF, AJG 2星)。

**核心命题**: Agent diversity是multi-agent LLM预测成功的必要条件。diversity取决于数据分离、角色强化、模型能力三个设计选择。

## Key Architecture

### Prediction Target
前瞻性20日已实现波动率 (forward-looking realized volatility):
`fwd_rv_20d = std(log_returns[t+1:t+21]) * sqrt(252)`

### Two-Stage Framework

**Stage 1: Multi-Agent Debate (v3, 当前版本)**
- 7个Specialist Agents: geopolitical, macro_demand, monetary, supply_opec, technical, sentiment, cross_market
- 3轮Debate: R1独立预测 → R2看到他人后修正 → R3再修正
- 每个agent只看自己领域数据(无common snapshot)
- System prompt含analytical bias (hawkish/contrarian/mean-reverting等)
- 输出: 各agent的per-round adjustment、confidence、behavior标签、influence graph
- API: gpt-5.4 via akane (sk-0cD1 key), temperature=0.5, call_delay=0.5s (RPM=150限制)

**Stage 2: Learned Sparse GAT Meta-Aggregation** (代码: `src/leak_free_gat.py`)

架构细节 (OptimizedGATModel, 最终版):
- **Graph Structure**: Learnable edge logits, sigmoid gating + DropEdge 20%
- **Role Embedding**: 每个agent学习4-dim identity vector
- **GNN Backbone**: 2层 Multi-Head GAT (4 heads, hidden_dim=24, top-k=3 per node)
- **Normalization**: Skip connection + LayerNorm after each GAT layer
- **Readout**: Mean pooling (flatten readout实测没帮助)
- **Output**: Regime-gated dual head (base_head + regime_head + gate)
- **Residual**: scaled by 0.05
- **Final**: pred = clamp(sum(weights * base_preds) + residual, min=0.05)

Leak-free node features (T1, 6-dim per agent): adj_r3, conf_r3, total_revision, conf_trend, revision_r23, agg_weight

Context features (14-dim): persist_vol, har_vol, total_adj, n_herding/7, vol_regime/3, debate_vol, persist-har gap, vol_change_5d, debate-har gap, mean_adj_r3, std_adj_r3, mean_conf_r3, std_conf_r3, mean_total_revision

Base predictions (3-dim, ridge-anchored): oof_ridge_pred, debate_vol, persist_vol

Training: Adam lr=0.003, weight_decay=1e-4, cosine LR, 250 epochs, grad clip=1.0

Label Embargo: 20天 (target label在t+20才完全观测)

Evaluation: Walk-forward (min_train=252d, retrain_every=63d), 5-seed ensemble, Newey-West HAC DM test (Bartlett kernel, bandwidth=19)

### Naive Aggregation
`pred = persistence_vol + confidence_weighted_mean(agent_adjustments)`

## Data

### Sources
- `data/oil_macro_daily.csv`: WTI价格、VIX、DXY、利率、波动率指标 (2011-01 ~ 2025-06)
- `data/gdelt_daily_features.csv`: GDELT地缘政治事件特征 (2013-04 ~ 2026-03)

### Results
- `results/debate_eval_full_v3_20260330_0319.csv`: v3 gpt-5.4全样本 (完成, n=2776)
- `results/debate_eval_full_v3_checkpoint.csv`: v3 checkpoint (同上)
- `results/debate_eval_full_v2_20260328_2014.csv`: v2 haiku全样本 (n=2776, 完成)
- `results/debate_eval_full_20260320_2343.csv`: v1 Gemini 3 Flash (n=1285, legacy)
- `results/leak_free_gat_results.csv`: v2 leak-free实验结果
- `results/leak_free_gat_v3_results.csv`: v3 leak-free实验结果 (全样本+优化)
- `results/debate_diversity_test_v3.csv`: v3 diversity测试 (29天)
- `results/vol_baselines_full.csv`: ML baselines
- `results/vol_baselines_dl_rolling.csv`: DL baselines

### Known Data Issues
- 2020-04-20 WTI负油价 (-$36.98) 污染前后各约20天的波动率计算
- 分析时需排除 actual_vol > 2 或 persist_vol > 2
- `_error` 列存储的是**平方误差** `(pred - actual)^2`
- debate CSV需sort_values('date')

## Key Scripts

### Core System
- `src/debate_system.py`: DebateEngine, SingleAgentBaseline, AttributionEngine
  - v3配置: gpt-5.4, akane API (sk-0cD1 key), 优化prompt, 数据分离, temperature=0.5
  - call_delay=0.5s (RPM=150限制), 空回重试15次
- `src/run_debate_eval.py`: 全样本评估, 支持 `--mode full/test`, `--tag v3`, `--rounds 3`, `--resume`
- `src/debate_diversity_test.py`: agent多样性对比测试

### Leak-Free Meta-Aggregation
- `src/leak_free_gat.py`: **当前主实验脚本** (feature diagnostic → Ridge → MLP → GAT), 支持v2/v3数据切换
- `src/walk_forward_utils.py`: embargo逻辑

### Legacy (v1, Shapley leakage contaminated)
- `src/optimized_gat.py`: v1 GAT (Shapley node features, leakage)
- `src/further_optimization.py`: v1优化变体 (DropEdge等)
- `src/final_comparison.py`: v1 GAT vs MLP + edge stability

### Baselines
- `src/run_vol_baselines.py`: ML baselines (Ridge/Lasso/GBR/RF)
- `src/run_vol_baselines_dl_rolling.py`: DL baselines (XGBoost/LSTM/Transformer)

## Running Evaluations

```bash
cd /Users/mac/computerscience/17Agent可解释预测

# v3 full debate (已完成)
# nohup /opt/miniconda3/bin/python -u src/run_debate_eval.py \
#     --mode full --tag v3 --horizon 20 --rounds 3 --attrib-samples 500 --resume \
#     > results/debate_v3_run.log 2>&1 &

# Leak-free GAT experiments (含优化版)
/opt/miniconda3/bin/python -u src/leak_free_gat.py

# ML/DL baselines
/opt/miniconda3/bin/python src/run_vol_baselines.py
/opt/miniconda3/bin/python src/run_vol_baselines_dl_rolling.py
```

## Project History

### v1: Gemini 3 Flash + Shapley features (2026-03-20, INVALIDATED)
- n=1285 (2020-01 ~ 2025-05), 2轮debate
- DropEdge GAT RMSE=0.1171, p=0.001 ← **Shapley look-ahead leakage**
- Shapley(t)使用actual_vol(t)计算，actual_vol(t)依赖未来收益率t+1:t+20
- 所有leak-free配置RMSE~0.15，和HAR持平
- 详见 docs/v2_leakage_analysis_20260328.md

### v2: Claude Haiku + per-agent features (2026-03-28, 完成但agent不够diverse)
- n=2776 (2013-04 ~ 2025-05), 3轮debate, claude-haiku
- Leak-free结果:
  - Ridge-debate_only: RMSE=0.1457, p=0.001 (最好)
  - T2-GAT: RMSE=0.1504, p=0.027
  - Per-agent features无增量信号 (partial r = -0.019)
- 根因: **Agent homogeneity**
  - 18.6%天7个agent给出完全相同的adj
  - R3 pairwise r=0.935
  - Herding率54.3%
- 详见 results/leak_free_gat_results.csv

### v3: GPT-5.4 + 优化prompt + 数据分离 (2026-03-29, debate完成, GAT实验完成)
- n=2776 (同样本), gpt-5.4, 优化prompt, 数据分离
- Debate完成: 39844 API calls, 99.6% success, 46.7M tokens
- 最终CSV: `results/debate_eval_full_v3_20260330_0319.csv`
- **Agent diversity确认提升8x** (29天测试):
  - R1 pairwise r: 0.675→-0.040 (接近独立)
  - R1 cross-agent std: 0.0055→0.0453 (8.2x)
  - R3保持多样(不收敛): 0.0016→0.0478 (29.9x)
- **v3 全样本 leak-free GAT结果 (n=2715, walk-forward n=2463)**:
  - Feature diagnostics:
    - std_adj_r3 vs actual_vol: r=+0.4987 (最强单特征)
    - sentiment adj_r3: r=+0.3479, geopolitical: r=+0.3497
    - Incremental R² (debate_vol + new features): +0.0697
  - Ridge results (all significant vs HAR=0.1600):
    - Ridge-cross_agent: RMSE=0.1404, p<0.0001 ***
    - Ridge-full_peragent: RMSE=0.1436, p=0.0008 ***
    - Ridge-debate_only: RMSE=0.1445, p=0.0005 ***
  - GAT results:
    - **T1-GAT: RMSE=0.1460, p=0.0075 *** (vs HAR)**
    - **T2-GAT: RMSE=0.1460, p=0.0092 *** (vs HAR)**
  - MLP results (not significant):
    - T2-MLP: RMSE=0.1508, p=0.1097
    - T1-MLP: RMSE=0.1530, p=0.2139
  - Pairwise:
    - **T1 GAT vs MLP: p=0.030 ** (图结构有增量, 5%显著)**
    - T2 GAT vs MLP: p=0.123
  - Baselines: HAR=0.1600, Persistence=0.1653, GARCH=0.4489, Naive debate=0.1648
  - 关键发现: MLP(flatten features)不显著，GAT(graph interaction)显著beat HAR
  - **优化后GAT (Grid-h24, hidden=24, ridge-anchored): RMSE=0.1418, p<0.0001**
  - GAT-Ridge差距从0.0056缩小到0.0014 (0.1418 vs 0.1404)
  - 优化关键: Ridge OOF prediction作为第三个base_pred是最大提升 (0.1460→0.1421)
  - Flatten readout没有帮助 (0.1466 vs 0.1460)，mean pooling + ridge anchoring是最优组合
- v3 prompt核心改进:
  1. 每个agent只看自己领域数据
  2. System prompt含analytical bias
  3. 辩论鼓励维持分歧
  4. Temperature 0.3→0.5

## 统计检验关键发现
- h=20前瞻性窗口导致forecast errors高度自相关 (ACF lag1=0.938)
- Naive DM test高估显著性约65%。必须用Newey-West HAC (Bartlett kernel, bandwidth=19)
- HAC校正后embargo修正后所有ML/DL baselines不再显著优于HAR
- Regime定义: fixed thresholds (0.20/0.35/0.55)

## TODO

### 当前进行中
- [x] v3 diversity测试 (29天, 确认8x提升)
- [x] v3全量debate (2776天完成, 39844 API calls)
- [x] v3全样本 leak-free GAT (n=2463, GAT p=0.008, GAT vs MLP p=0.030)
- [x] GAT优化 (0.1460→0.1418, 逼近Ridge 0.1404, p<0.0001)
- [ ] 论文重写 (基于v3 clean results)

### 论文写作
- manuscript/v1/: 存档v1文稿 (contaminated numbers)
- 需要基于v3结果重写所有sections
- 目标: IJF, 结构待v3结果后重新规划

## Documentation
- `docs/v3_results_summary_20260330.md`: **v3最终结果总结 (组会汇报用)**
- `docs/v2_leakage_analysis_20260328.md`: Shapley泄露分析报告
- `docs/research_framework.md`: 研究框架
- `docs/methodology_framework.md`: 文献锚定的9步方法论
- `docs/embargo_revalidation_20260324.md`: Label embargo修正报告
- `docs/literature_scan_2024_2026.md`: 文献扫描
- `docs/archive_v1v2/`: v1时代的旧文档和图表 (已归档)
- `bib/references.bib`: 参考文献库
- `outline_ijf.md`: IJF版论文大纲 (需基于v3更新)
- `figures/sdd_gat_framework_v3.svg`: v3框架图
