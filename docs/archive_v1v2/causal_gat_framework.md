# Causal Graph Attention Network for Multi-Agent Debate Aggregation

## Motivation

原始debate系统使用confidence-weighted mean聚合agent adjustments，存在两个问题:
1. 聚合权重仅依赖agent自报的confidence，不考虑agent间交互结构
2. 在高波动率regime和herding严重时预测崩溃 (RMSE从0.14升至0.33)

通过分析发现:
- Herding agent数量与预测误差强正相关
- 0个herding时debate大幅赢HAR; 7个全herding时debate崩溃
- Debate在low/normal/elevated regime都赢HAR，仅在high regime输

## Framework Architecture

### Stage 1: Causal Discovery

对7个agent的Shapley值时间序列进行Granger因果检验:
- 检验 agent_i 是否Granger-cause agent_j (lag=5, alpha=0.005)
- 构建因果邻接矩阵，区分真实信息传递 vs 虚假herding关联
- 每次retrain时在训练集上重新估计因果图

Granger因果检验公式:
```
Restricted:  y_j(t) = Σ_k β_k * y_j(t-k) + ε
Unrestricted: y_j(t) = Σ_k β_k * y_j(t-k) + Σ_k γ_k * y_i(t-k) + ε
F-test on RSS reduction
```

### Stage 2: Graph Attention Network

**Node Features (7维 per agent):**
- Shapley value (当日)
- Myerson value (当日)
- Behavior encoded (herding=0, anchored=1, independent=2, overconfident=3)
- Graph degree (influence graph度数)
- Shapley MA5 (5日移动平均)
- Shapley STD5 (5日标准差)
- Herding streak (近5日herding次数)

**Context Features (9维):**
- persist_vol, har_vol, debate_vol
- total_adj (debate - persistence)
- n_herding / 7 (herding agent比例)
- vol_regime (0-3)
- persist-har gap
- vol_change_5d (5日波动率变化)
- debate-har gap

**Model Architecture:**
```
Input: node_feats (batch, 7, 7) + adj (7, 7) + context (batch, 9)
  → GAT Layer 1: (7, 7) → (7, 16), LeakyReLU attention
  → ReLU
  → GAT Layer 2: (7, 16) → (7, 16)
  → ReLU
  → Flatten: (7*16) = 112
  → Concat with context: 112 + 9 = 121
  → Weight Head: 121 → 32 → 16 → 5 (softmax weights)
  → Residual Head: 121 → 16 → 1 (* 0.05 scaling)

Output: pred = Σ(w_i * base_i) + residual
  where base = [debate_vol, har_vol, persist_vol, single_vol, garch_vol]
```

### Training Protocol

- Walk-forward evaluation: 不使用未来数据
- min_train = 252 days (1 year)
- retrain_every = 63 days (quarterly)
- Optimizer: Adam (lr=0.003, weight_decay=1e-4)
- LR Schedule: Cosine annealing
- Epochs: 200 per retrain window
- Early stopping via best checkpoint

## Results

### Overall Performance (n=972 walk-forward eval days)

| Model | RMSE | MAE | DM vs HAR |
|-------|------|-----|-----------|
| **Causal-GAT** | **0.1368** | **0.1072** | **-9.825*** |
| Simple Blend | 0.1501 | 0.1162 | -5.977* |
| Debate (original) | 0.1512 | 0.1166 | -3.050* |
| HAR | 0.1549 | 0.1181 | baseline |
| Persistence | 0.1551 | 0.1174 | 0.188 |

### By Volatility Regime

| Regime | Causal-GAT | HAR | Debate | n |
|--------|-----------|-----|--------|---|
| Low | 0.1509 | 0.2004 | 0.1950 | 65 |
| Normal | 0.1407 | 0.1479 | 0.1405 | 454 |
| Elevated | 0.1252 | 0.1362 | 0.1327 | 377 |
| High | **0.1550** | 0.2232 | 0.2334 | 76 |

Causal-GAT在所有regime都优于HAR，特别是high regime改善幅度最大 (30.6%)。

### Learned Weight Patterns

模型自动学到了regime-dependent的组合策略:

| Regime | Debate | HAR | Persist | Single | GARCH |
|--------|--------|-----|---------|--------|-------|
| Low | 16.5% | 34.6% | 5.3% | 43.4% | 0.2% |
| Normal | 14.8% | 38.2% | 5.4% | 41.3% | 0.2% |
| Elevated | 13.0% | 36.4% | 3.4% | 47.0% | 0.2% |
| High | 17.9% | **46.0%** | 3.3% | 32.7% | 0.1% |

关键发现:
1. GARCH被完全抛弃 (权重<0.3%)
2. Single agent获得最高权重 (43%)，说明多agent debate的herding反而伤害了聚合质量
3. 高波动时HAR权重自动升高 (46%)，模型学会了在危机期更信任统计模型
4. Persistence权重很低 (4.5%)，表明加工过的预测优于raw persistence

## Key Insights for Paper

1. **Naive aggregation is the bottleneck**: confidence-weighted mean丢弃了丰富的agent交互信息
2. **Herding detection enables improvement**: 因果发现能区分真实信息传递和herding，为聚合提供结构先验
3. **Regime-adaptive weighting**: 不同市场环境下最优的预测组合不同，GNN自动学习了这种适应性
4. **Single agent paradox**: multi-agent debate产生了有价值的信息(Shapley值均为负)，但naive聚合反而不如single agent; Causal-GAT通过学习权重解决了这个问题
