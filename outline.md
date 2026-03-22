# 论文结构草案 v2

## Title
Emergent Accountability in Multi-Agent AI Systems: A Myerson Value Attribution Framework for Collective Decision Failures

---

## 1. Introduction

1. 组织正从单AI工具走向多Agent AI系统辅助决策（Hughes et al. 2025; Bommarito et al. 2025）
2. 多Agent系统的问责挑战：当集体决策失灵时，谁该负责？
3. 现有算法问责理论止步于单agent（Zeiser 2024的attributability gap）
4. 提出「涌现问责缺口」：agent交互产生的问责失灵，是单agent问责缺口的指数级放大
5. 本文贡献：四维归因框架（WHO/WHEN/WHAT/HOW）+ Myerson value路径级归因 + 原油市场实证

## 2. Theoretical Background

### 2.1 Algorithmic Accountability: From Single to Multi-Agent
- 单agent问责的三层次：attributability, answerability, accountability (Zeiser 2024)
- 多agent场景下的新挑战：涌现性、路径依赖、责任稀释
- 管理学文献缺口确认（Jarrahi & Ritala 2025 最接近但仅概念层面）

### 2.2 Multi-Agent AI Systems in Organizations
- AI作为组织行动者（Anthony et al. 2023; Stelmaszak et al. 2025）
- 多Agent风险分类学（Hammond et al. 2025）
- 从principal-agent到principal-multi-agent的治理转变

### 2.3 Collective Intelligence and Information Cascades
- 集体智慧的条件（多样性、独立性、分权）
- 信息级联和羊群效应在AI agent中的表现（TwinMarket, Yang et al. 2025）
- 辩论作为促进/抑制集体智慧的双刃剑

### 2.4 Cooperative Game Theory for Attribution
- Shapley value基础与性质
- Myerson value：图限制下的Shapley扩展
- 从ML特征归因到MAS行为归因的演进

## 3. Conceptual Framework

### 3.1 Emergent Accountability Gap
- 正式定义
- 与单agent attributability gap的关系
- 涌现条件和边界

### 3.2 Four-Dimensional Accountability Framework
- WHO / WHEN / WHAT / HOW
- 每个维度对应的组织治理决策
- HOW维度的创新性论证

### 3.3 Research Questions and Propositions

## 4. Research Design

### 4.1 Design Science Approach
- 为什么选择design science（构建artifact来验证理论命题）
- artifact = 多agent辩论预测系统 + Myerson归因框架

### 4.2 Empirical Context: Oil Market Crisis Forecasting
- 为什么选择油价（多驱动因素、频繁regime切换、极端事件丰富）
- 数据描述

## 5. System Design: Multi-Agent Debate Architecture

### 5.1 Agent Design
- 三个专业化analyst agents + aggregator
- 异质信念设计的理论基础（Hawkish-Dovish, Takano et al. 2025）

### 5.2 Structured Debate Protocol
- 多轮辩论机制
- 结构化输入输出格式

### 5.3 Debate Influence Graph Construction
- 节点和边的定义
- 影响权重的量化方法

## 6. Attribution Methodology

### 6.1 Counterfactual Simulation
- 反事实基线定义
- 替换策略

### 6.2 Shapley Value Attribution (Baseline)
- 公式化
- Monte Carlo近似

### 6.3 Myerson Value Attribution (Proposed)
- 图限制联盟
- 路径级归因的实现
- 计算方法（Tarkowski et al. 2020）

### 6.4 Multi-Dimensional Aggregation
- Agent维度、时间维度、行为维度聚合
- 可视化设计

### 6.5 Targeted Intervention
- 基于归因分数的定向干预机制
- 与随机干预的对比

## 7. Experimental Design

### 7.1 Walk-Forward Evaluation Setup
### 7.2 Baselines
### 7.3 Evaluation Metrics
### 7.4 Ablation Studies

## 8. Results

### 8.1 Prediction Performance
- 多Agent辩论 vs 单Agent vs 无Agent基线

### 8.2 Attribution Analysis
- 四维归因结果
- Shapley vs Myerson对比
- 极端事件case study（负油价、俄乌冲突、中东升级）

### 8.3 Emergent Behaviors
- 羊群效应的量化证据
- 锚定效应和过度自信传染
- 辩论改善vs恶化决策的边界条件

### 8.4 Intervention Effectiveness
- 定向干预 vs 随机干预 vs 全体干预
- 干预效率的Pareto分析

## 9. Discussion

### 9.1 Theoretical Implications
- 涌现问责缺口的实证支持
- 对算法问责理论的扩展
- 对组织信息处理理论的贡献

### 9.2 Practical Implications
- 多Agent系统的组织治理指南
- agent角色设计、权限分配、通信架构的归因驱动优化
- 监管视角

### 9.3 Limitations and Future Research
- LLM依赖性和可重复性
- 样本量和极端事件稀缺
- 扩展到其他决策领域

## 10. Conclusion
