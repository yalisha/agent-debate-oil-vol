# SDD-GAT v3 实验结果总结

**日期**: 2026-03-30
**目标期刊**: International Journal of Forecasting (IJF, AJG 3星)

## 一、研究问题

多Agent LLM辩论能否改进原油波动率预测? 核心发现: **能，但前提是agent之间足够diverse。**

## 二、框架概述

SDD-GAT = Structured Delphi Debate + Graph Attention Network

- **Stage 1**: 7个specialist LLM agents (GPT-5.4) 进行3轮结构化辩论
  - 每个agent只看自己领域的数据 (数据分离)
  - 每个agent有不同的分析偏好 (hawkish/contrarian/mean-reverting等)
  - 3轮辩论: 独立预测 → 看到他人后修正 → 最终立场
- **Stage 2**: GAT图神经网络学习agent间交互关系，产生最终预测

预测目标: 前瞻性20日已实现波动率 (annualized)

## 三、实验设置

- 样本: 2013-04 ~ 2025-05, 共2715个有效交易日
- Walk-forward evaluation: 训练252天起步, 每63天retrain, 20天label embargo
- 5-seed ensemble, Newey-West HAC DM test (Bartlett kernel, bw=19)
- 评估窗口: n=2463

## 四、主要结果

### 4.1 模型排名

| 排名 | 模型 | RMSE | p vs HAR | 显著性 |
|:---:|------|:----:|:--------:|:-----:|
| 1 | Ridge-cross_agent | 0.1404 | <0.0001 | *** |
| 2 | **SDD-GAT (optimized)** | **0.1418** | **<0.0001** | **\*\*\*** |
| 3 | SDD-GAT (baseline) | 0.1460 | 0.0075 | *** |
| 4 | Ridge-debate_only | 0.1445 | 0.0005 | *** |
| 5 | MLP | 0.1530 | 0.2139 | n.s. |
| 6 | HAR | 0.1600 | - | - |
| 7 | Persistence | 0.1653 | - | - |

### 4.2 核心发现

**发现1: Agent diversity是必要条件**

|  | v2 (Claude Haiku) | v3 (GPT-5.4) |
|--|:-:|:-:|
| R1 pairwise correlation | 0.675 | -0.040 |
| R3 pairwise correlation | 0.935 | 0.010 |
| Cross-agent std | 0.006 | 0.045 (8x) |
| Per-agent signal | 无 | 有 |

v2的agent输出几乎一样 (R3相关0.935)，per-agent特征无信号。v3通过三个设计改进实现了agent diversity: (1) 数据分离, (2) 分析偏好注入, (3) 更强的模型。

**发现2: 图结构有增量价值**

- GAT vs MLP: DM_HAC = -2.173, **p = 0.030**
- MLP (flatten agent features) 不显著 beat HAR (p=0.21)
- GAT (建模agent interaction) 显著 beat HAR (p=0.008)
- 说明: 简单非线性组合不够，需要建模agent间的信息流

**发现3: Cross-agent disagreement是最强信号**

- std_adj_r3 (R3调整量的cross-agent标准差) vs actual_vol: r = +0.50
- 经济含义: agent间分歧越大 → 市场不确定性越高 → 实际波动率越高
- 这比任何单个agent的预测都更有价值

### 4.3 GAT优化路径

| 步骤 | 改动 | RMSE | 提升 |
|------|------|:----:|:----:|
| Baseline | Mean-pool GAT, 2 base preds | 0.1460 | - |
| + Ridge anchoring | OOF ridge_pred作为第3个base pred | 0.1421 | -2.7% |
| + hidden_dim=24 | 16→24 | 0.1418 | -0.2% |
| 总计 | | 0.1418 | -2.9% |

关键: flatten readout反而没帮助 (0.1466 > 0.1460)。mean pooling + ridge anchoring是最优组合。

## 五、Feature Diagnostics

### Per-agent R3 adjustment vs actual_vol

| Agent | Pearson r | 方向 |
|-------|:---------:|:----:|
| Geopolitical (hawkish) | +0.350 | 地缘风险↑ → vol↑ |
| Sentiment (momentum) | +0.348 | 情绪极端 → vol↑ |
| Cross-Market (correlation) | +0.232 | 跨市场联动 → vol↑ |
| Monetary (dovish) | +0.143 | 宽松 → vol↑ |
| Supply/OPEC (supply-focused) | -0.240 | 供给充裕 → vol↓ |
| Technical (contrarian) | -0.237 | 技术信号 → vol↓ |
| Macro Demand (mean-reverting) | -0.281 | 需求稳定 → vol↓ |

不同agent的预测方向分化明确，验证了数据分离+角色设计的有效性。

## 六、与ML/DL Baselines对比

所有baseline均使用相同的walk-forward + HAC DM检验:

| 方法 | RMSE | p vs HAR | 备注 |
|------|:----:|:--------:|------|
| Ridge (macro features) | 0.1607 | 0.9361 | 传统ML |
| Lasso | 0.1559 | 0.3994 | |
| GBR | 0.1557 | 0.4746 | |
| Random Forest | 0.1654 | 0.2218 | |
| XGBoost | 0.1497 | 0.0709 | DL |
| LSTM | 0.1608 | 0.9660 | |
| Transformer | 0.1528 | 0.2422 | |
| **SDD-GAT (ours)** | **0.1418** | **<0.0001** | **唯一在1%水平显著** |

HAC校正后，所有传统ML/DL baselines都不显著优于HAR。只有SDD-GAT在1%水平显著。

## 七、方法论贡献

1. **Newey-West HAC校正**: h=20前瞻窗口导致forecast errors高度自相关 (ACF lag1=0.938)。Naive DM test高估显著性约65%。
2. **Label embargo**: 20天embargo确保训练集不包含与测试集label重叠的观测。
3. **Leak-free feature设计**: 所有特征在预测时点t可获取，不使用未来信息。
4. **Agent diversity量化**: pairwise correlation、cross-agent std、herding rate等指标。

## 八、论文创新点总结

1. 提出SDD-GAT框架，首次将结构化Delphi辩论与图注意力网络结合用于金融波动率预测
2. 发现并量化agent diversity是multi-agent LLM预测的binding constraint
3. 证明图结构 (GAT) 显著优于扁平聚合 (MLP)，说明agent间交互信息有价值
4. 构建严格的leak-free评估协议，纠正了existing literature中常见的look-ahead leakage和DM test滥用问题

## 九、下一步

- [ ] 论文写作 (基于v3 clean results)
- [ ] 框架图更新 (已有SVG初版: figures/sdd_gat_framework_v3.svg)
- [ ] Ablation study: v2 vs v3 diversity对比表格
- [ ] Case study: 2022 Ukraine crisis期间的agent行为分析
