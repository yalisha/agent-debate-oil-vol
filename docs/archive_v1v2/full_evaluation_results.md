# 全样本日频评估结果

评估时间: 2026-03-20
数据范围: 2020-01-02 至 2025-06-12，共1273个交易日

## 1. 整体预测精度

| 方法 | 类型 | RMSE | MAE | 相对Debate |
|------|------|------|-----|-----------|
| **Debate (7-agent)** | LLM辩论 | **0.0424** | 0.0182 | 基准 |
| Persistence | 统计 | 0.0436 | 0.0173 | +2.8% |
| Single-agent | 单LLM | 0.0611 | 0.0296 | +44.1% |
| HAR | 统计 | 0.0655 | 0.0299 | +54.5% |
| Lasso | ML (daily retrain) | 0.0716 | 0.0283 | +68.9% |
| Ridge | ML (daily retrain) | 0.0725 | 0.0446 | +71.0% |
| GBR | ML (daily retrain) | 0.0929 | 0.0384 | +119.1% |
| GARCH(1,1) | 统计 | 0.1716 | 0.0913 | +304.7% |
| Transformer | DL (季度retrain) | 0.1816 | 0.0601 | +328.3% |
| XGBoost | ML (季度retrain) | 0.1853 | 0.0456 | +337.0% |
| RF | ML (daily retrain) | 0.1799 | 0.0590 | +324.3% |
| LSTM | DL (季度retrain) | 0.2005 | 0.0733 | +372.9% |
| HistMean | 统计 | 0.3164 | 0.1628 | +646.2% |

注: ML (daily retrain) 使用252天滚动窗口每日重训; DL (季度retrain) 每63个交易日重训一次，使用所有历史数据。

## 2. 分Regime表现

### Debate系统及统计Baseline

| Regime | n | Debate | Persist | Single | HAR | GARCH |
|--------|---|--------|---------|--------|-----|-------|
| Low (<20%) | 74 | 0.0239 | 0.0250 | 0.0265 | 0.0248 | 0.0842 |
| Normal (20-35%) | 571 | 0.0232 | 0.0239 | 0.0253 | 0.0326 | 0.0751 |
| Elevated (35-55%) | 501 | 0.0383 | 0.0397 | 0.0466 | 0.0559 | 0.0999 |
| Crisis (>55%) | 126 | 0.0976 | 0.0998 | 0.1343 | 0.1585 | 0.4649 |

### ML Baseline (daily retrain)

| Regime | n | Lasso | Ridge | GBR | RF |
|--------|---|-------|-------|-----|-----|
| Low | 74 | 0.0337 | 0.0631 | 0.0276 | 0.0360 |
| Normal | 571 | 0.0290 | 0.0446 | 0.0310 | 0.0304 |
| Elevated | 501 | 0.0430 | 0.0556 | 0.0586 | 0.0857 |
| Crisis | 126 | 0.1054 | 0.1596 | 0.2599 | 0.5235 |

### DL Baseline (季度 rolling retrain, 每63天重训)

| Regime | n | XGBoost | LSTM | Transformer |
|--------|---|---------|------|-------------|
| Low | 74 | 0.0247 | 0.0343 | 0.0349 |
| Normal | 571 | 0.0297 | 0.0542 | 0.0405 |
| Elevated | 501 | 0.0465 | 0.0774 | 0.0703 |
| Crisis | 126 | 0.5664 | 0.6067 | 0.5525 |

## 3. 核心发现

### 发现1: Debate系统在所有regime上RMSE最优或接近最优
- Low vol: GBR(0.028)略优于Debate(0.024)，但差距很小
- Normal/Elevated/Crisis: Debate均为最优

### 发现2: Persistence是极强的baseline
- 所有ML方法（Ridge, Lasso, GBR, RF）整体RMSE都打不过简单的Persistence
- 这是波动率预测领域的经典现象: vol clustering使得昨天的值本身就是很好的预测
- Debate系统的优势在于: 在persistence基础上做了正确的边际调整

### 发现3: Crisis期分化最大
- Debate vs Single-agent: 0.098 vs 0.134（27.3%改善）
- Debate vs GBR: 0.098 vs 0.260（62.3%改善）
- Debate vs RF: 0.098 vs 0.524（81.3%改善）
- ML方法在crisis期普遍崩溃，因为训练数据中crisis样本稀少

### 发现4: Debate比Single-agent稳定
- Single-agent在所有regime都比Debate差
- 差距在crisis期最大（0.134 vs 0.098）
- 多agent辩论通过多元视角互相制衡，抑制了单agent的极端预测

## 4. 行为偏差分布（n=1273）

| 行为 | 频率 | 说明 |
|------|------|------|
| Independent（独立判断） | 60.0% | Agent在Round 2维持或强化独立观点 |
| Herding（羊群效应） | 27.1% | Agent在Round 2向多数意见靠拢 |
| Anchored（锚定） | 12.3% | Agent在Round 2几乎不修正初始预测 |

### 归因模式

| 最常被归因的Agent | 频率 |
|-------------------|------|
| Geopolitical | 48.2% |
| Sentiment | 14.8% |
| Cross-market | 11.7% |

## 5. LLM使用统计

- 总调用次数: 17,792
- 失败次数: 30（成功率99.8%）
- 总token数: 16,546,541
- 估算成本: 约41元

### 发现5: DL方法在crisis期全面崩溃
- XGBoost/LSTM/Transformer在低波动期表现尚可（0.025-0.035），但crisis期RMSE飙到0.55-0.61
- 即使用季度rolling retrain也无法解决，因为crisis本身就是稀有事件
- Debate系统crisis期RMSE仅0.098，是DL方法的1/6

## 6. 待补充

- [x] XGBoost, LSTM, Transformer baseline
- [ ] Intervention experiment结果需要修正（均值被outlier污染，应用trimmed mean）
- [ ] Case study分析
- [ ] 行为偏差的regime-dependent回归
