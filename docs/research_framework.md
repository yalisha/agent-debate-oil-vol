# 多Agent LLM辩论系统中的群体行为偏差：来自原油波动率预测的实证证据

## 一、研究问题

**核心问题：** LLM agents在多轮辩论中是否表现出系统性的群体行为偏差？这些偏差如何随市场不确定性变化？它们对集体预测质量有什么影响？

**子问题：**

- RQ1: 多Agent辩论相比单Agent，在不同市场状态下对预测质量的影响是否存在异质性？
- RQ2: LLM agents在辩论中是否表现出可量化的羊群效应、锚定效应和过度自信？
- RQ3: 这些行为偏差的强度如何随市场不确定性水平变化？
- RQ4: 能否通过合作博弈论方法（Shapley value）将集体预测误差归因到具体agent和行为模式？

## 二、理论基础

### 2.1 信息级联理论（Banerjee 1992; Bikhchandani et al. 1992）

个体理性地忽略私有信息、跟随前人行动，导致信息级联。Banerjee (1992, QJE) 的模型预测: 不确定性越高，agent越倾向于放弃私有信息。Pitre et al. (2025, ACL Findings) 将sycophancy识别为LLM debate中信息级联的算法机制。

### 2.2 集体智慧与Delphi方法（Surowiecki 2004; Dalkey & Helmer 1963）

集体智慧依赖多样性、独立性、去中心化和有效聚合四个条件。Delphi方法通过匿名迭代和可控反馈实现这些条件。Lorenz & Fritz (2026) 证明LLM可替代人类专家执行Delphi流程，达到人类面板水平的校准性 (r = 0.87-0.95)。Schoenegger et al. (2024, Science Advances) 发现12个LLM的集成预测精度接近925名人类预测者。

### 2.3 异质Agent模型（Hommes 2006; Kukacka & Barunik 2014）

Hommes (2006) 的HAM框架证明agent之间的信息异质性产生现实市场动态。Kukacka & Barunik (2014, Physica A) 实证研究herding、overconfidence和sentiment如何在HAM中影响市场波动。Axtell & Farmer (2025, JEL) 的权威综述确立了agent-based modeling在经济学中的地位。

### 2.4 LLM群体行为偏差（Ashery et al. 2025; Madigan et al. 2025）

Ashery et al. (2025, Science Advances) 发现LLM群体自发涌现集体偏差，即使个体无偏差。Madigan et al. (2025) 在金融决策中观测到multi-agent系统的涌现偏差无法追溯到个体。Agarwal & Khanna (2025) 的CW-POR指标揭示debate中说服力压倒事实的风险。这些发现支撑了在真实预测任务中系统研究LLM群体偏差的必要性。

## 三、方法设计: Structured Delphi Debate (SDD)

> 详细文献锚定见 `docs/methodology_framework.md`，以下为精要版。

### 3.1 预测目标: 前瞻性已实现波动率

$$RV_{t \to t+20} = \sqrt{252} \cdot \text{std}(r_{t+1}, \ldots, r_{t+20})$$

选择已实现波动率而非GARCH条件方差，依据 Andersen & Bollerslev (1998) 和 Corsi (2009) 确立的范式。选择WTI原油是因为其波动率受多维信息驱动（Kilian 2009; Herrera et al. 2018），天然适合multi-agent信息分解。

### 3.2 信息维度分解 → 7个Specialist Agent

7个agent对应油价波动率的7个信息维度，设计依据Kilian (2009) 的结构性冲击分解和Hommes (2006) 的异质agent框架:

| Agent | 信息维度 | 理论依据 |
| ----- | ------- | ------- |
| Geopolitical | 地缘政治风险 | Caldara & Iacoviello (2022); Nonejad (2021) |
| Macro Demand | 宏观需求 | Hamilton (2003); Kilian (2009) |
| Monetary | 货币政策 | Kilian & Zhou (2022) |
| Supply/OPEC | 供给侧 | Kilian (2009) 供给冲击识别 |
| Technical | 量化/统计 | Corsi (2009) HAR异质时间尺度 |
| Sentiment | 新闻情绪 | Dai et al. (2026); Hashami & Maldonado (2025) |
| Cross-Market | 跨市场传染 | Diebold & Yilmaz (2012) 波动率溢出 |

### 3.3 辩论协议: LLM-Delphi

框架定位为 **LLM-powered Structured Delphi**，依据Dalkey & Helmer (1963) 和 Lorenz & Fritz (2026):

1. **Round 1 (Independent Elicitation)**: 7个agent基于各自信息独立预测，对应Delphi第一轮
2. **Round 2 (Informed Revision)**: 每个agent看到其他6个的预测和证据后修正，对应Delphi的controlled feedback

两轮设计的依据: Du et al. (2023, ICML) 发现2-3轮辩论最优; Liang et al. (2023, EMNLP) 发现过多轮次导致思维退化; Pitre et al. (2025, ACL) 发现sycophancy随轮次增加而恶化。

### 3.4 Persistence + Adjustment 预测架构

$$\hat{\sigma}_{t \to t+20} = \sigma_{t-20 \to t} + \frac{\sum_{i=1}^{7} c_i \cdot \Delta_i}{\sum_{i=1}^{7} c_i}$$

- $\sigma_{t-20 \to t}$: 当前已实现波动率 (persistence baseline)，利用volatility clustering (Mandelbrot 1963; Engle 1982)
- $\Delta_i \in [-0.5, +0.5]$: Agent $i$ 的调整量预测
- $c_i$: Agent $i$ 的自报置信度

LLM预测adjustment而非绝对值，因为Tan et al. (2024, NeurIPS) 发现LLM在纯数值预测上无优势，但Williams et al. (2024) 证明LLM在context-rich场景表现优越。纯数学聚合依据Bates & Granger (1969) forecast combination理论和Hanea et al. (2021) 的质量加权方法。

### 3.5 行为偏差识别

基于Round 1→Round 2的变化自动分类agent行为:

- **Herding**: 向中位数方向移动 (Banerjee 1992; Pitre et al. 2025 sycophancy机制)
- **Anchoring**: 几乎不修正Round 1预测 (Tversky & Kahneman 1974)
- **Overconfidence**: 大幅调整但保持高置信度 (Moore & Healy 2008)
- **Independent**: 基于证据做出非趋同调整

LLM群体偏差的实证支撑: Ashery et al. (2025, Science Advances) 发现LLM群体自发涌现集体偏差; Madigan et al. (2025) 在金融MAS中观测到不可追溯到个体的涌现偏差。

### 3.6 Shapley归因

将7-agent聚合建模为合作博弈 $(N, v)$，$v(S) = (y_t - \hat{\sigma}_S)^2$。依据 Shapley (1953) 的四公理唯一解，Monte Carlo近似 (M=500排列)。$\phi_i < 0$ 表示改善预测，$\phi_i > 0$ 表示恶化。

应用依据: Xia et al. (2025) 的HiveMind在multi-agent LLM系统中使用Shapley值做贡献归因; Bimonte et al. (2024) 用Shapley值为预测ensemble分配权重。

### 3.7 Baseline体系与评估

四层12个baseline:

- **持续性**: Persistence (random walk benchmark)
- **经典计量**: HAR (Corsi 2009), GARCH(1,1) (Bollerslev 1986)
- **机器学习**: Ridge, Lasso, GBR, RF (Christensen et al. 2023), XGBoost (Tiwari et al. 2024)
- **深度学习**: LSTM (Ben Romdhane & Boubaker 2026), Transformer (Qiu et al. 2025)
- **LLM消融**: Single-Agent (同信息无辩论，隔离debate增量价值，对应Du et al. 2023)

评估: Diebold-Mariano检验 (HAC lag=20)，分4个regime (Low/Normal/Elevated/Crisis)

## 四、初步实证结果

### 4.1 预测精度（月频采样，n=61，覆盖2020-2025全周期）

| 方法 | RMSE | 相对Debate |
|------|------|-----------|
| **Debate（7-agent辩论）** | **0.0345** | **最优** |
| Persistence | 0.0359 | +4.1% |
| HAR | 0.0432 | +25.2% |
| Single-agent | 0.0561 | +62.6% |
| GARCH(1,1) | 0.2042 | +491.9% |

### 4.2 分Regime表现

| Regime | n | Debate | Single | HAR | Persist | Debate vs Single |
|--------|---|--------|--------|-----|---------|-----------------|
| Low vol | 4 | 0.017 | 0.014 | 0.021 | 0.020 | 1.2x worse |
| Normal | 22 | **0.011** | 0.017 | 0.017 | 0.011 | **1.6x better** |
| Elevated | 29 | 0.047 | 0.041 | 0.053 | 0.050 | 0.9x (接近) |
| Crisis | 6 | **0.028** | 0.150 | 0.064 | 0.017 | **5.5x better** |

**核心发现：** 多Agent辩论的价值在危机期最大。单agent在Crisis下严重高估波动率（RMSE=0.15），7-agent辩论通过多元观点互相制衡控制在0.028。

### 4.3 行为模式分布

| 行为 | 频率 |
|------|------|
| Independent（独立判断） | 47.5% |
| Herding（羊群效应） | 36.1% |
| Anchored（锚定） | 16.4% |

36%的辩论中出现羊群效应，agents在看到同行观点后系统性地向多数意见靠拢。

### 4.4 归因模式

| 最常被归因的Agent | 频率 |
|-------------------|------|
| Geopolitical | 49.2% |
| Sentiment | 16.4% |
| Technical | 14.8% |

地缘政治分析师最常成为预测偏差的主要来源。

### 4.5 待完善

- 全样本日频评估（n=1273）正在运行中，将提供更可靠的统计推断
- 干预实验需要重新设计
- 行为分类方法需要更严格的效度验证
- 需要case study分析（COVID负油价、俄乌冲突、2025中东升级等）

## 五、创新点

### 创新1：多Agent LLM辩论行为偏差的大规模实证

现有研究：
- Bini et al. (NBER 2026): 单agent，实验室任务
- Herd Behavior paper (2025): multi-agent，抽象任务
- Science Advances (2025): LLM群体涌现，社会规范场景

**我们的增量：** 在真实金融预测任务中，基于4800+次LLM调用，定量记录7个agent在辩论中的行为偏差模式。首次将LLM behavioral bias从实验室搬到真实决策场景。

### 创新2：行为偏差的regime-dependent特征

多agent辩论的效果因市场状态而异：危机期5.5倍优于单agent，但低波动期略差。这意味着辩论机制不是简单地「更好」或「更差」，而是通过改变信息聚合方式，在高不确定性下激活集体智慧、在低不确定性下引入冗余噪声。

### 创新3：基于Shapley归因的偏差溯源

不只是发现偏差存在，还能溯源到具体agent。Shapley value提供了「谁是偏差的主要来源」的定量回答，为组织层面的AI系统治理提供可操作的工具。

## 六、文献支撑

### 直接底子论文

| 论文 | 发表 | 与本研究的关系 |
|------|------|---------------|
| Bini, Cong, Huang & Jin (2026) | NBER WP / NeurIPS 2025 | 单agent LLM behavioral bias的系统实证。我们扩展到multi-agent交互 |
| "Herd Behavior in LLM-based MAS" (2025) | OpenReview | 最直接前序：证明LLM agent存在peer-induced herding。但在抽象任务上 |
| Agiza et al. (2025) "Emergent social conventions and collective bias in LLM populations" | Science Advances | LLM群体自发产生集体偏差。顶刊认可，证明方向成立 |
| Lou & Sun (2025) "Anchoring bias in LLMs" | J. Computational Social Science | LLM锚定效应的实证证据 |
| "LLMs are overconfident and amplify human bias" (2025) | ArXiv | LLM过度自信且放大人类偏差 |
| Fed FEDS 2025-090 | Federal Reserve | GenAI在金融市场中的herding风险和金融稳定影响 |

### 管理学/IS文献支撑

| 论文 | 发表 | 提供的支撑 |
|------|------|-----------|
| Jarrahi & Ritala (2025) | California Management Review | Multi-agent complexity as governance challenge |
| Mikalef et al. (2025) | EJIS editorial | 呼吁artifact-centered responsible AI研究 |
| Hillebrand, Raisch & Schad (2025) | AMJ Annals | 人-AI协作的控制-问责对齐框架 |
| Zeiser (2024) | Science & Engineering Ethics | 单agent attributability gap，我们论证在multi-agent下放大 |
| Hughes et al. (2025) | JCIS | 50位专家共识：multi-agent AI重塑行业决策 |
| Hammond et al. (2025) | Cooperative AI Foundation | Multi-agent risk taxonomy |

### 方法论文献

| 论文 | 发表 | 提供的方法 |
|------|------|-----------|
| Du et al. (2024) | ICML 2024 | 多agent辩论的方法论基础 |
| Tang et al. (2026) | ArXiv | MAS极端事件Shapley归因框架 |
| Banerjee (1992) | QJE | 信息级联理论 |
| Corsi (2009) | J. Financial Econometrics | HAR模型（baseline） |

### Gap确认

**MISQ、ISR、Management Science、Organization Science、AMJ、AMR中，没有论文在真实金融任务中实证研究多Agent LLM的群体行为偏差。**

## 七、目标期刊

| 优先级 | 期刊 | 理由 |
|--------|------|------|
| 1 | Management Science | "AI for Finance and Business Decisions"虚拟特刊，明确欢迎algorithmic bias研究 |
| 2 | ISR | "GenAI and New Methods of Inquiry"特刊（deadline 2026.09） |
| 3 | MISQ | "AI-IA Nexus"特刊 |
| 备选 | CMR | Agentic AI专题，已有8+篇2025-2026 |
| 会议 | ICIS 2026 | Working paper投递 |

## 八、研究路线图

### Phase 1：实证基础（当前阶段）
- [x] 构建7-agent辩论系统
- [x] 月频采样验证（n=61）
- [ ] 全样本日频评估（n=1273，运行中）
- [ ] Case study分析（COVID、俄乌、中东升级）

### Phase 2：深化分析
- [ ] 行为偏差的regime-dependent回归分析
- [ ] 行为分类的robustness check（不同阈值、不同分类方法）
- [ ] Shapley归因的统计显著性检验
- [ ] 信息级联的动态分析（偏差如何在轮次间传播）

### Phase 3：理论构建与写作
- [ ] 理论框架：信息级联 + 集体智慧条件 + AI behavioral science
- [ ] 实证结果整理与可视化
- [ ] 讨论：对组织AI治理的启示
- [ ] 投稿Management Science或ISR
