# Proposal v2: Emergent Accountability in Multi-Agent AI Decision Systems

## 一句话定位

当组织部署多Agent AI系统辅助决策时，个体agent的合理行为可能在交互中涌现出集体失灵。本研究提出基于辩论影响图的Myerson归因框架，实现对集体决策失灵的路径级问责，并以原油市场危机预测为实证平台进行验证。

## 英文标题

**Emergent Accountability in Multi-Agent AI Systems: A Myerson Value Attribution Framework for Collective Decision Failures**

*Subtitle: Evidence from Oil Market Crisis Forecasting*

---

## 1. 研究背景与问题

### 1.1 实践背景

组织正在从部署单一AI工具转向部署多Agent AI系统辅助决策。在金融分析、风险评估、战略规划等场景中，多个专业化AI agent协同工作已成趋势（Hughes et al. 2025; Bommarito et al. 2025）。然而，当这些系统集体做出错误决策时，一个根本性问题浮现：**谁该为集体失灵负责？**

### 1.2 理论缺口

现有算法问责理论（Zeiser 2024; Collina et al. 2023）聚焦于单一AI系统：一个AI给出一个建议，人类决策者接受或拒绝。但多Agent系统的问责问题本质不同：

- **涌现性**：集体失灵可能源自agent间的交互，而非任何单个agent的错误
- **路径依赖**：agent A的输出影响agent B的判断，影响沿图结构传播
- **责任稀释**：多个agent共同参与使得「谁该负责」变得模糊

Zeiser (2024) 提出单agent场景下的 attributability gap。本研究认为，**在多agent系统中，这一缺口因交互效应而呈指数级放大**，产生我们称之为 **涌现问责缺口 (Emergent Accountability Gap)** 的现象。

### 1.3 文献确认

经系统检索，**目前top管理学/IS期刊（MISQ、ISR、Management Science、Organization Science、AMJ、AMR）中没有论文专门理论化多Agent AI系统在组织决策中的问责问题**（详见 bib/05）。Jarrahi & Ritala (2025, CMR) 最为接近，承认multi-agent complexity是挑战，但停留在概念层面。Mikalef et al. (2025) 在EJIS发editorial明确呼吁对agentic AI artifact做研究。

### 1.4 研究问题

**主RQ：当组织部署多Agent AI系统辅助决策时，如何实现对集体决策失灵的有效问责？**

- RQ1: 集体决策失灵能否被形式化地归因到具体的agent、时间点、行为类型和影响路径？
- RQ2: 多Agent辩论在什么条件下改善、在什么条件下恶化决策质量？
- RQ3: 图限制归因（Myerson value）相比扁平归因（Shapley value）是否更能识别失灵根源？
- RQ4: 基于归因的定向干预能否有效降低危机期系统风险？

---

## 2. 理论框架

### 2.1 核心概念：涌现问责缺口

**定义：** 当多个AI agent在交互过程中产生了任何单个agent均未预期也未直接导致的决策失灵时，现有问责机制无法有效归因的状态。

**与existing theory的关系：**
- 扩展 Zeiser (2024) 的 attributability gap：从单agent到多agent
- 连接 Hammond et al. (2025) 的技术风险分类学与组织治理理论
- 回应 Mikalef et al. (2025) 对 artifact-centered responsible AI 研究的呼吁

### 2.2 理论锚定

| 理论 | 在本研究中的角色 |
|------|------------------|
| 算法问责 (Algorithmic Accountability) | 核心框架：从单agent扩展到多agent |
| 组织信息处理理论 (Galbraith 1974) | 解释多agent辩论如何处理环境不确定性 |
| 集体智慧条件 (Surowiecki 2004) | 多样性、独立性、分权：何时辩论改善vs恶化决策 |
| 信息级联理论 (Bikhchandani et al. 1992) | 解释agent间羊群效应和过度自信传染 |

### 2.3 四维问责框架

本研究提出四维归因框架用于实现多Agent系统的结构化问责：

| 维度 | 问题 | 方法 | 管理意义 |
|------|------|------|----------|
| **WHO** | 哪个agent责任最大 | 按agent聚合归因值 | agent权限与角色设计 |
| **WHEN** | 哪个时间点最关键 | 按时间/辩论轮次聚合 | 干预时机与预警 |
| **WHAT** | 哪种行为最危险 | 按行为类型聚合（羊群、锚定、过度自信） | 行为约束与防护 |
| **HOW** | 影响如何传播 | Myerson value沿图路径归因 | 通信架构与隔离设计 |

第四维HOW是本研究的方法论核心创新，利用Myerson value实现路径级归因。

---

## 3. 方法论

### 3.1 多Agent辩论预测系统

#### Agent设计

三个专业化analyst agents + 一个aggregator：

| Agent | 角色 | 信息源 | 信念倾向 |
|-------|------|--------|----------|
| 地缘政治分析师 | 评估地缘风险对油价的影响 | GDELT事件流、冲突指标 | 倾向关注供给侧风险 |
| 宏观经济师 | 评估需求面与政策环境 | VIX、DXY、利率、PMI | 倾向关注需求面基本面 |
| 技术分析师 | 识别价格动量与结构 | 价格序列、波动率、均线 | 倾向关注短期趋势信号 |
| 聚合器 | 整合三方观点给出最终预测 | 三个analyst的结构化输出 | 加权聚合 |

#### 辩论机制

参考 Hawkish-Dovish (Takano et al. 2025) 的信念修正模式：

```
Round 1: 各agent独立分析 → 输出结构化观点 {方向, 强度, 置信度, 证据}
Round 2: 各agent看到其他agent的Round 1输出 → 修正或坚持 → 输出修正后观点 + 修正理由
Round 3: 同上
Final:   聚合器综合三轮辩论 → 输出最终预测 + 聚合理由
```

每轮输出均结构化为JSON，确保可追溯和可量化。

### 3.2 辩论影响图 (Debate Influence Graph)

**定义：** 有向加权图 G = (V, E, W)

- **节点** V = {(agent_i, round_r) | i ∈ {geo, macro, tech}, r ∈ {1,2,3}}
  - 每个节点携带属性：观点方向、强度、置信度
- **边** E = {((agent_i, round_r), (agent_j, round_{r+1})) | i ≠ j}
  - 当且仅当 agent_j 在round_{r+1}的观点相比round_r发生了可度量的变化
- **权重** W(e) = agent_j 观点变化幅度中可归因于 agent_i 输出的比例
  - 量化方法：比较agent_j在 {有agent_i输出} vs {无agent_i输出} 条件下的观点差异

每个预测时点 t 生成一张辩论影响图 G_t。

### 3.3 反事实归因计算

#### Shapley Value（基准）

对于预测时点 t 的预测误差 L_t，agent i 的 Shapley 归因值为：

$$\phi_i(L_t) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [L_t(S \cup \{i\}) - L_t(S)]$$

其中 L_t(S) 是只包含agent集合 S 参与时的预测误差，通过反事实模拟获得：将不在 S 中的agent替换为中性baseline（预测「方向不明，置信度0」）。

Monte Carlo近似：采样M=1000个排列，每个排列依次加入agent计算边际贡献。

#### Myerson Value（核心方法）

Myerson value 在 Shapley 基础上增加图限制：只有在图 G_t 中相连的agent子集才构成有效联盟。

$$\mu_i(L_t, G_t) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} \sum_{C \in \mathcal{C}(S \cup \{i\})} [L_t(C) - L_t(C \setminus \{i\})]$$

其中 C(S) 是 S 在图 G_t 中的连通分量集合。

计算方法参考 Tarkowski et al. (2020) 的 hybrid Monte Carlo algorithm。

#### 三维聚合

- **Agent维度**：Φ_agent(i) = Σ_t φ_i(L_t)，识别系统性高风险agent
- **时间维度**：Φ_time(t) = Σ_i φ_i(L_t)，识别风险积聚的关键时间点
- **行为维度**：将agent行为分类为 {独立判断, 跟随他人, 锚定初始观点, 过度自信}，按类型聚合

### 3.4 定向干预验证

基于归因结果进行干预：

1. 识别top-k高归因agent-时间组合
2. 替换这些agent的行为为safe baseline（中性预测）
3. 重新运行聚合，比较干预前后的预测误差

与随机干预、全体干预等基线对比，验证定向干预的效率。

---

## 4. 实证设计

### 4.1 数据

| 数据 | 来源 | 用途 |
|------|------|------|
| WTI原油日频价格 | EIA, 2020-01 至 2026-03 | 预测目标 |
| 宏观指标 | FRED: VIX, DXY, YIELD_10Y | 宏观agent输入 |
| GDELT地缘事件 | GDELT 2.0 | 地缘agent输入 |
| 价格技术指标 | 计算衍生 | 技术agent输入 |

样本期覆盖：2020-04负油价、2022俄乌冲突、2023-2026中东局势升级等多个极端事件。

### 4.2 评估框架

**Walk-forward设计**：252个交易日训练窗口，逐日滚动预测。

**预测评估指标**：
- RMSE, MAE
- Directional Accuracy
- 极端事件窗口的分regime表现

**归因评估指标**：
- Pareto集中度（Gini系数）：归因是否集中于少数agent/时点
- 路径一致性：Myerson归因路径是否与辩论记录中的实际影响一致
- 干预效率：定向干预vs随机干预的误差降低比
- Shapley vs Myerson对比：图限制归因是否提供更精确的失灵定位

### 4.3 对比基线

| 基线 | 说明 |
|------|------|
| 单Agent系统 | 现有 agent_framework.py 的单LLM编排器 |
| 多Agent无辩论 | 各agent独立预测，简单平均聚合 |
| 多Agent辩论 + 无归因 | 有辩论但无Shapley/Myerson归因 |
| 多Agent辩论 + Shapley | 有辩论 + 扁平Shapley归因（无图限制） |
| 多Agent辩论 + Myerson | 本文方法（辩论 + 图限制归因） |

### 4.4 消融实验

1. 移除辩论机制（agent独立预测 → 聚合）
2. 移除图结构约束（Myerson退化为Shapley）
3. 移除agent异质性（同质prompt）
4. 改变辩论轮数（1轮 vs 2轮 vs 3轮 vs 5轮）
5. 改变agent数量（2 vs 3 vs 5 vs 7）

---

## 5. 预期贡献

### 5.1 理论贡献

1. **涌现问责缺口概念**：提出并实证验证多Agent AI系统中accountability gap的涌现性质，扩展Zeiser (2024)的单agent理论
2. **四维归因框架 (WHO/WHEN/WHAT/HOW)**：为组织部署多Agent AI系统提供结构化问责工具。HOW维度（路径级归因）是方法论创新
3. **多Agent辩论的双刃剑效应**：实证识别辩论改善vs恶化决策的边界条件，连接集体智慧与信息级联理论

### 5.2 方法论贡献

1. **辩论影响图 + Myerson归因**：首次将Myerson value应用于LLM多agent辩论系统的路径级极端事件归因
2. **从合成到真实**：在真实金融数据上验证Tang et al. (2026) 在合成环境中发现的五个MAS极端事件规律

### 5.3 实践贡献

1. **组织AI治理指南**：归因结果可直接指导agent角色设计、权限分配、通信架构
2. **定向干预机制**：基于归因分数的精准干预，比全局干预更高效

---

## 6. 与现有文献的区分

### 6.1 vs Tang et al. (2026)

| 维度 | Tang et al. | 本研究 |
|------|-------------|--------|
| 归因方法 | Shapley（扁平） | Myerson（图限制） |
| 环境 | 合成模拟 | 真实金融数据 |
| Agent类型 | 简单行为规则 | LLM驱动辩论 |
| 归因维度 | 3维（agent/time/behavior） | 4维（+影响路径） |
| 理论框架 | AI safety | 组织问责 |

### 6.2 vs HiveMind DAG-Shapley (AAAI 2026)

| 维度 | HiveMind | 本研究 |
|------|----------|--------|
| 图结构 | DAG（无环） | 一般图（辩论有环） |
| 目标 | prompt优化 | 极端事件归因 |
| 归因方法 | DAG-Shapley | Myerson value |

### 6.3 vs TradingAgents (ICML 2025)

| 维度 | TradingAgents | 本研究 |
|------|---------------|--------|
| 可解释性 | 自然语言透明性 | 形式化归因 + 自然语言 |
| 问责 | 无 | 四维归因框架 |
| 极端事件 | 未特别关注 | 核心焦点 |

---

## 7. 目标期刊

### 首选
- **MISQ** (Management Information Systems Quarterly): IS领域top期刊，design science + behavioral研究均接受
- **ISR** (Information Systems Research): 关注IT artifact的组织影响
- **JSIS** (Journal of Strategic Information Systems): Papagiannidis et al. (2025) 已在此发responsible AI governance

### 备选
- **TMIS** (ACM Transactions on Management Information Systems): HAD论文已发此刊
- **Management Science**: 接受computational + empirical组合
- **California Management Review**: Jarrahi & Ritala (2025) 已在此讨论multi-agent

### 会议备选
- **ICIS / ECIS**: IS顶会，可先投working paper
- **AAMAS**: multi-agent系统顶会
- **ACM ICAIF**: AI+Finance交叉

---

## 8. 实施计划

### Phase 1: 多Agent辩论系统构建
- 基于现有agent_framework.py扩展为3+1 agent架构
- 实现结构化辩论协议（JSON输入输出）
- 实现辩论影响图的自动构建

### Phase 2: 归因框架实现
- 实现反事实模拟引擎
- 实现Shapley value的Monte Carlo近似
- 实现Myerson value的图限制归因
- 实现三维聚合和可视化

### Phase 3: 实证实验
- 全样本walk-forward预测
- 极端事件窗口deep-dive
- 消融实验
- Shapley vs Myerson对比

### Phase 4: 论文撰写
- 理论框架与文献综述
- 方法论描述
- 实证结果与讨论
- 管理启示

---

## 9. 核心参考文献

| Paper | Role |
|-------|------|
| Tang et al. (2026) arXiv:2601.20538 | 直接基础：MAS极端事件Shapley归因 |
| Zeiser (2024) Science & Eng. Ethics | 理论基础：attributability gap |
| Jarrahi & Ritala (2025) CMR | 最接近的管理学文献 |
| Mikalef et al. (2025) EJIS editorial | 学术信号：呼吁agentic AI研究 |
| Hammond et al. (2025) arXiv:2502.14143 | 技术风险分类学：multi-agent failure modes |
| Angelotti (2023) KBS | Myerson value用于MAS解释的先例 |
| Chen et al. (2018) ICLR 2019 | C-Shapley = Myerson的理论连接 |
| HiveMind (2025) AAAI 2026 | DAG-Shapley在LLM agent中的先例 |
| Takano et al. (2025) PRIMA | 信念驱动辩论的方法论参考 |
| TradingAgents (2024) ICML 2025 | 多agent金融辩论的代表性工作 |
| Tarkowski et al. (2020) arXiv:2001.00065 | Myerson value的Monte Carlo计算 |
| Du et al. (2024) ICML | 多agent辩论的方法论基础 |
