# 多Agent辩论系统预测机制

## 预测目标

明日20天年化实现波动率（annualized realized volatility），基于 walk-forward 评估，无 look-ahead。

## 核心公式

```
σ̂_{t+1} = σ_t + Σ(w_i × Δ'_i)
```

- `σ_t`: 今日20天实现波动率（persistence baseline）
- `Δ'_i`: 第 i 个 agent 在 Round 2 的波动率调整量预测
- `w_i`: 基于 confidence 的归一化权重

## 预测流程

### Step 1: 数据准备

每个交易日准备两类输入:

**公共市场快照（所有agent共享）**
- WTI 价格、收益率、20日/60日波动率
- VIX、DXY、美债收益率、利差
- 动量指标

**Agent 专属数据（按角色分发）**

| Agent | 专属信息 |
|-------|---------|
| 地缘政治分析师 | GDELT冲突指数、油区国家事件频率、物质冲突占比 |
| 宏观需求分析师 | 上证指数、工业产出代理变量 |
| 货币政策分析师 | Fed Funds Rate、实际收益率、收益率曲线 |
| OPEC供给分析师 | 库存代理变量、供给中断信号 |
| 技术分析师 | 历史波动率模式、GARCH特征、均值回归信号 |
| 新闻情绪分析师 | GDELT 新闻量、媒体tone、舆论热度 |
| 跨市场传染分析师 | VIX变化、股商相关性、信用利差 |

### Step 2: Round 1 独立预测

7个agent并行调用LLM（GPT-5.4, temperature=0.3），各自根据角色定义和专属数据，输出结构化JSON:

```json
{
  "vol_adjustment": +0.03,
  "direction": "up",
  "confidence": 0.7,
  "evidence": ["GDELT conflict intensity up 2 std", "Gulf tensions escalating"]
}
```

`vol_adjustment = +0.03` 表示预测波动率比昨天高3个百分点。

Prompt 中明确提示: "most days vol changes by less than 0.02. Only predict large adjustments (>0.05) when evidence is strong." 这限制了 agent 过度调整的倾向。

### Step 3: Round 2 辩论修正

每个agent收到其他6个agent的预测摘要和证据，格式如:
```
[geopolitical] adj=+0.030 (up), conf=0.70 | GDELT conflict up; Gulf tensions
[technical] adj=-0.010 (down), conf=0.60 | Vol mean reversion signal; GARCH decay
...
```

Agent 被指示: "Review the other analysts' forecasts. Revise your view if their evidence is compelling, or hold your position if your analysis is stronger."

### Step 4: 数学聚合

不再调用LLM，直接用纯数学聚合:

```python
weighted_adj = Σ(confidence_i × adjustment_i) / Σ(confidence_i)
final_vol = max(persistence_vol + weighted_adj, 0.05)
```

### 设计理由

1. **Persistence + Adjustment 架构**: 波动率高度自相关（vol clustering），昨天的值本身就是很强的预测。系统把 persistence 内建，LLM 只需判断边际变化
2. **纯数学聚合**: 避免再加一次LLM调用引入噪声，也便于 Shapley 归因的反事实计算
3. **并行调用**: 同一轮内的 agent 用线程池并行调用LLM，减少延迟
4. **调整量 clamp**: `max(-0.5, min(0.5, adj))`，防止LLM输出极端值

## 与Baseline的区别

| 方法 | 预测方式 |
|------|---------|
| Persistence | σ̂ = σ_t（直接用昨天的值） |
| HAR | σ̂ = c + β₁σ_t + β₅σ_t^(w) + β₂₂σ_t^(m)（线性回归日/周/月波动率） |
| GARCH(1,1) | 条件方差模型，基于历史收益率序列拟合 |
| ML (Ridge/GBR/RF) | 用宏观+技术+GDELT特征回归预测波动率 |
| Single-Agent | 单个LLM看所有数据，直接输出波动率预测 |
| **Debate (7-Agent)** | **7个专业LLM各自预测调整量，辩论修正，加权聚合** |
