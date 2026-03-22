# Gap Analysis and Research Positioning

Literature survey conducted 2026-03-18.

---

## 1. What Already Exists

### Multi-Agent LLM Financial Prediction
- TradingAgents, FinCon, QuantAgents, MASFIN 等已经展示了多 agent 协作预测的可行性
- Bull/Bear debate (TradingAgents), belief-based debate (Hawkish-Dovish) 已有先例
- 但这些系统的**可解释性主要依赖自然语言透明性**，缺乏形式化的归因工具

### Shapley Value for MAS
- Tang et al. (2026) 建立了 MAS 极端事件的 Shapley 归因框架，但**仅在合成模拟环境**测试
- MARL 领域有大量 Shapley credit assignment 工作 (SHAQ, Shapley Q-value)，但面向**训练时信用分配**，非事后解释
- HiveMind (AAAI 2026) 提出了 DAG-Shapley 用于 LLM agent workflow，但**面向 prompt 优化**，非极端事件归因

### Graph-Structured Agent Communication
- G-Designer, GTD, AMAS 研究了 agent 通信拓扑的动态设计
- 但这些工作聚焦于**优化通信效率**，未将图结构用于归因和解释

### Oil/Commodity Forecasting with LLM
- Beyond Polarity (2026) 用 SHAP 分析 LLM 情感特征对 WTI 的贡献
- Ghali et al. (2025) 用 agentic AI 提取新闻信号预测商品价格冲击
- **但没有任何工作将多 agent 辩论机制用于原油预测**

---

## 2. Identified Gaps

### Gap 1: 辩论交互 + 形式化归因的缺失
**现状：** 现有多 agent 金融系统要么有辩论但无形式化归因 (TradingAgents, FactorMAD)，要么有 Shapley 归因但无 agent 交互 (Tang et al. 在合成环境)。
**机会：** 将 Shapley/Myerson 归因框架应用于有真实辩论交互的金融预测系统，实现路径级归因。

### Gap 2: 合成环境 vs 真实金融数据
**现状：** Tang et al. 的极端事件归因框架仅在 EconAgent, TwinMarket, SocialNetwork 三个合成环境验证。合成-真实的 gap 是 MAS 研究公认的 open problem。
**机会：** 在真实油价数据 + 真实地缘政治事件 (GDELT) 上验证 Shapley 归因的五个发现是否成立。

### Gap 3: 图结构用于归因 vs 用于通信优化
**现状：** G-Designer/GTD/AMAS 将图结构用于优化 agent 通信拓扑。Angelotti (2023) 将 Myerson value 用于 MAS 解释但在简单环境。
**机会：** 将辩论过程的影响传播建模为有向图，用 Myerson value 做图限制归因，实现**路径级极端事件解释**。这在现有文献中没有先例。

### Gap 4: LLM Agent 群体行为偏差的实证研究
**现状：** TwinMarket 展示了 LLM agent 的羊群效应和过度自信，但是在交易模拟中。没有工作系统研究 LLM analyst agents 在辩论-预测场景中的群体行为偏差。
**机会：** 利用 Shapley 归因量化 LLM agents 在油价极端事件中的羊群效应、锚定效应、过度自信传染等行为偏差。连接行为金融学理论。

### Gap 5: 商品/能源市场的多 agent 预测系统
**现状：** 几乎所有多 agent LLM 预测系统都面向股票市场。原油/大宗商品领域只有单 agent 或非 agent 方法。
**机会：** 首个面向原油市场的多 agent LLM 辩论预测系统，利用 GDELT 地缘事件数据作为独特信息源。

---

## 3. Our Unique Positioning

```
                    形式化归因
                        ↑
                        |
    Tang et al.         |         [OUR WORK]
    (Shapley +          |         (Myerson value +
     合成模拟)           |          真实金融数据 +
                        |          辩论图结构)
                        |
   ─────────────────────┼─────────────────────→ Agent 交互
                        |                        (辩论/图)
                        |
    SHAP/feature        |         TradingAgents
    attribution         |         FactorMAD
    (无 agent 交互)      |         (辩论但无形式化归因)
                        |
```

### 核心贡献定位

1. **方法论：** 提出基于辩论影响图的 Myerson value 归因框架，实现路径级极端事件解释。这是对 Tang et al. (2026) 从扁平 Shapley 到图限制 Shapley 的方法论扩展。

2. **实证：** 在真实原油市场极端事件（伊朗危机、俄乌冲突等）上验证 MAS 极端事件归因的五个规律是否在真实金融数据上成立，弥合合成-真实的 gap。

3. **发现：** 系统量化 LLM analyst agents 在危机辩论中的群体行为偏差（羊群、锚定、过度自信传染），建立与行为金融学的理论连接。

4. **应用：** 基于归因分数的定向干预机制，验证在真实金融预测场景中能否有效降低危机期预测风险。

---

## 4. Key Papers to Cite as Foundation

| Paper | Why |
|-------|-----|
| Tang et al. (2026) arXiv:2601.20538 | 直接基础：MAS 极端事件 Shapley 归因框架 |
| Du et al. (2024) arXiv:2305.14325 | 多 agent 辩论方法论基础 |
| HiveMind (2025) arXiv:2512.06432 | DAG-Shapley 在 LLM agent 中的先例 |
| Angelotti (2023) arXiv:2212.03041 | Myerson value 用于 MAS 解释的先例 |
| Chen et al. (2018) arXiv:1808.02610 | C-Shapley = Myerson value 的理论连接 |
| TradingAgents (2024) arXiv:2412.20138 | 多 agent 金融辩论的代表性工作 |
| Hawkish-Dovish (2025) arXiv:2511.02469 | 信念驱动辩论在经济预测中的应用 |
| TwinMarket (2025) arXiv:2502.01506 | LLM agent 涌现行为的金融市场证据 |
| SHAQ (2022) arXiv:2105.15013 | Markov Shapley value 的理论基础 |
| Tarkowski et al. (2020) arXiv:2001.00065 | Myerson value 的 Monte Carlo 计算方法 |

---

## 5. Potential Target Venues

- **AI:** NeurIPS, ICML, ICLR, AAAI, AAMAS
- **Finance + AI:** ACM ICAIF, FinNLP Workshop
- **Information Systems:** TMIS, MISQ
- **Computational Finance:** Journal of Computational Finance, Quantitative Finance
