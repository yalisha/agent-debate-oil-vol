# Methodology Framework: Literature-Grounded Design

## Overview

本文提出 **Structured Delphi Debate (SDD)** 框架，将多个专业化LLM agent组织为类Delphi专家面板，通过结构化辩论预测原油波动率，并在此过程中识别和分析群体行为偏差。以下每个设计步骤均有明确的文献支撑。

---

## Step 1: 预测目标选择 — 已实现波动率 (Realized Volatility)

### 设计
预测目标为未来20个交易日的年化已实现波动率:

$$RV_{t \to t+20} = \sqrt{252} \cdot \text{std}(r_{t+1}, r_{t+2}, \ldots, r_{t+20})$$

其中 $r_t = \ln(P_t / P_{t-1})$ 为对数收益率。

### 文献支撑

**为什么用Realized Volatility而非GARCH条件方差:**
- Andersen & Bollerslev (1998) 证明，用日内高频数据构造的已实现波动率是真实波动率的一致估计量，远优于GARCH条件方差作为预测评估标准
- Corsi (2009) 的HAR-RV模型奠定了以已实现波动率为目标的预测范式，成为该领域标准benchmark

**为什么选20天窗口:**
- Corsi (2009) 的HAR模型使用日(1d)、周(5d)、月(22d)三个时间尺度，20d接近月度窗口，捕捉中频波动率动态
- Christensen et al. (2023, JFE) 在ML vs HAR对比中使用22天已实现波动率，发现ML在较长horizon上优势更明显

**为什么选WTI原油:**
- 原油是全球交易量最大的大宗商品，波动率受多维信息驱动（地缘政治、供需、货币政策、情绪），天然适合multi-agent信息分解
- Herrera et al. (2018, IJF) 系统比较了原油波动率预测方法，确立了该领域的评估标准
- Bouri et al. (2020) 和 Nonejad (2021) 证明地缘政治风险和疫情不确定性对原油波动率有显著预测力

---

## Step 2: 信息维度分解 — 7个Specialist Agent的理论基础

### 设计
将油价波动率的信息来源分解为7个正交维度，每个维度分配一个专业化LLM agent:

| Agent | 信息维度 | 数据源 | 理论根据 |
|-------|---------|--------|---------|
| Geopolitical | 地缘政治风险 | GDELT事件数据 | Caldara & Iacoviello (2022) GPR Index |
| Macro Demand | 宏观需求 | 上证指数、动量 | Hamilton (2003) 需求驱动油价 |
| Monetary | 货币政策 | Fed Funds、DXY、Yield curve | Kilian & Zhou (2022) 货币传导 |
| Supply/OPEC | 供给侧 | WTI价格、动量 | Kilian (2009) 供给冲击识别 |
| Technical | 技术/量化 | 历史波动率、收益率 | Corsi (2009) HAR异质agent |
| Sentiment | 新闻情绪 | GDELT tone、新闻量 | Dai et al. (2026) LLM多维情绪 |
| Cross-Market | 跨市场传染 | VIX、利差变化 | Diebold & Yilmaz (2012) 波动率溢出 |

### 文献支撑

**信息分解的理论根据:**
- Kilian (2009, AER) 将油价变动分解为供给冲击、需求冲击和投机冲击三个结构性来源，证明不同信息维度对油价的驱动机制不同
- Corsi (2009) 的HAR模型假设市场中存在异质性agent（日交易者、周交易者、月交易者），各自处理不同时间尺度的信息。我们的设计将这种异质性从时间维度扩展到信息维度
- Hommes (2006, Handbook of Computational Economics) 的异质agent模型(HAM)框架证明，agent之间的信息异质性是产生现实市场动态（excess volatility、fat tails、volatility clustering）的关键

**7个维度的具体来源:**
- 地缘政治: Caldara & Iacoviello (2022) 构建GPR指数，Nonejad (2021, FRL) 证明GPR对原油波动率有非线性预测力
- 宏观需求: Hamilton (2003) 和 Kilian (2009) 识别需求侧因素对油价波动的结构性贡献
- 货币政策: Kilian & Zhou (2022) 证明美元和利率通过金融渠道影响大宗商品波动率
- 供给/OPEC: Kilian (2009) 的供给冲击识别，OPEC决策是原油特有的供给侧不确定性来源
- 技术面: Corsi (2009) 和 Andersen et al. (2007) 证明历史波动率模式（clustering、mean reversion）包含显著预测信息
- 情绪: Dai et al. (2026) 用GPT-4o提取多维情绪信号预测WTI收益，发现intensity和uncertainty维度的预测力超越简单极性; Hashami & Maldonado (2025) 证明新闻量本身就是robust predictor
- 跨市场: Diebold & Yilmaz (2012) 的波动率溢出框架证明资产间存在显著的波动率传染效应

---

## Step 3: 辩论协议 — Structured Delphi Method

### 设计
两轮结构化辩论:
- **Round 1 (Independent Elicitation)**: 7个agent基于各自信息独立输出预测
- **Round 2 (Informed Revision)**: 每个agent看到其他6个agent的预测和证据后修正

### 文献支撑

**Delphi方法的理论传统:**
- Dalkey & Helmer (1963) 提出Delphi方法，通过匿名、迭代、可控反馈三个原则聚合专家判断，减少social pressure导致的偏差
- Lorenz & Fritz (2026) 的 Scalable Delphi 直接证明LLM可以替代人类专家执行Delphi流程，达到与人类专家面板相当的校准性（Pearson r = 0.87-0.95）
- Bertolotti & Mari (2025) 进一步验证LLM-Delphi在预测GenAI演化中的可行性

**为什么恰好2轮:**
- Du et al. (2023, ICML 2024) 的multi-agent debate实验发现，2-3轮辩论收敛效果最佳，更多轮次带来边际收益递减
- Liang et al. (2023, EMNLP) 发现过度辩论导致"Degeneration-of-Thought"（思维退化），agent陷入mutual reinforcement。2轮平衡了信息交换与独立性保持
- Pitre et al. (2025, ACL Findings) 的CONSENSAGENT研究发现sycophancy（谄媚性）随轮次增加而恶化，少量轮次有助于缓解

**Round 2的信息暴露设计:**
- 每个agent看到所有其他agent的预测值、置信度和核心证据，但不看到其他agent的系统prompt
- 这对应Delphi方法中的"controlled feedback"原则: 提供他人判断但不施加社会压力
- Chen et al. (2024, ACL) 的ReConcile框架证明，暴露他人的答案和推理可以提升multi-agent推理准确率最多11.4%

---

## Step 4: Persistence + Adjustment 预测架构

### 设计
最终预测 = 当前已实现波动率 + 加权调整量:

$$\hat{\sigma}_{t \to t+20} = \sigma_{t-20 \to t} + \frac{\sum_{i=1}^{7} c_i \cdot \Delta_i}{\sum_{i=1}^{7} c_i}$$

其中 $\sigma_{t-20 \to t}$ 为当前20日已实现波动率（persistence baseline），$\Delta_i$ 为agent $i$ 在Round 2的调整量预测，$c_i$ 为agent $i$ 的自报置信度。

Agent输出的是调整量 $\Delta_i \in [-0.5, +0.5]$，而非绝对波动率。

### 文献支撑

**为什么以Persistence为基线:**
- 波动率聚类（volatility clustering）是金融时序最稳健的经验事实之一（Mandelbrot, 1963; Engle, 1982），当前波动率是未来波动率的强预测因子
- Corsi (2009) 的HAR模型本质上是在persistence基础上叠加多时间尺度的修正项: $\hat{\sigma}_t = c + \beta_d \sigma_t^{(d)} + \beta_w \sigma_t^{(w)} + \beta_m \sigma_t^{(m)}$
- 我们的设计保持了这一思想: LLM agent不需要"从零预测"波动率，只需要基于新信息判断边际调整
- 这等价于文献中的 "forecasting innovation" 或 "error correction" 思路

**为什么预测调整量而非绝对值:**
- Tan et al. (2024, NeurIPS) 发现LLM在纯数值时序预测上并不比简单attention层更强，但Williams et al. (2024) 证明LLM在context-rich场景下表现优越
- 让LLM预测adjustment利用了LLM的context理解优势（判断地缘事件的波动率含义），同时避免了其数值预测的劣势
- Halawi et al. (2024, NeurIPS) 证明LLM更擅长基于信息做出定性/半定量判断（方向和幅度），而非精确数值预测

**Adjustment的范围限制 [-0.5, +0.5]:**
- 硬性clamp防止LLM输出极端值（hallucination），属于prompt engineering中的constrained generation
- 实际观测中日间波动率变化极少超过0.5（50个百分点），因此这是合理的物理约束

---

## Step 5: 置信度加权聚合 — Forecast Combination Theory

### 设计
使用confidence-weighted linear combination:

$$\hat{\Delta} = \frac{\sum_{i=1}^{7} c_i \cdot \Delta_i}{\sum_{i=1}^{7} c_i}$$

纯数学聚合，不再调用LLM。

### 文献支撑

**Forecast combination的理论基础:**
- Bates & Granger (1969) 奠基之作，证明forecast combination几乎总是优于单一最佳预测，因为即使最佳模型也不包含其他模型的全部信息
- Wang et al. (2023, IJF) 的50年综述指出，基于质量指标的差异化加权可以显著优于简单平均
- Gneiting & Ranjan (2013) 从预测校准性角度论证了加权聚合的数学性质

**为什么用Confidence加权:**
- Hanea et al. (2021, PLoS ONE) 证明，基于可测量的专家质量指标（推理质量、知识程度、信息敏感度）进行差异化加权，优于简单平均和中位数
- Karvetski et al. (2013, Decision Analysis) 提出概率一致性加权，证明内部一致性高的预测者应获得更高权重
- Chen et al. (2024, ACL) 在multi-agent LLM中直接使用confidence-weighted voting，实验验证了该方法在LLM场景下的有效性
- Yao et al. (2025) 的Roundtable Policy进一步证实confidence-weighted consensus在科学推理任务上的优势

**为什么是纯数学聚合而非LLM聚合:**
- 避免引入额外的LLM调用噪声
- 保证Shapley归因的反事实计算在数学上可精确执行（需要精确控制每个agent的inclusion/exclusion）
- Schoenegger et al. (2024, Science Advances) 发现简单平均12个LLM的预测就能达到接近人类群体的精度，支持机械聚合的有效性

---

## Step 6: 行为偏差识别 — 辩论过程中的Behavioral Classification

### 设计
基于Round 1和Round 2之间的变化，自动分类每个agent的行为:

| 行为类型 | 操作化定义 | 判定条件 |
|---------|-----------|---------|
| Herding (羊群效应) | Agent在Round 2向其他agent的中位数方向移动 | $\|\Delta_i^{R2} - \text{median}\| < \|\Delta_i^{R1} - \text{median}\|$ 且 shift > 阈值 |
| Anchoring (锚定效应) | Agent在Round 2几乎不修正Round 1的预测 | $\|\Delta_i^{R2} - \Delta_i^{R1}\| < \epsilon$ |
| Overconfidence (过度自信) | Agent在Round 2大幅调整但保持高置信度 | $c_i^{R2} > 0.85$ 且 shift > 阈值 |
| Independent (独立判断) | Agent基于证据做出非趋同的调整 | 不满足上述任一条件 |

### 文献支撑

**LLM表现出人类认知偏差:**
- Ashery et al. (2025, Science Advances) 在去中心化LLM群体中观测到自发涌现的集体偏差，即使个体agent无偏差
- Madigan et al. (2025) 发现金融决策中multi-agent系统存在无法追溯到个体的涌现偏差
- Agarwal & Khanna (2025) 提出CW-POR指标，量化debate中说服力压倒事实的概率，揭示LLM的overconfidence和susceptibility

**Herding的理论根基:**
- Banerjee (1992, QJE) 信息级联理论: 当agent理性地忽略私有信息、跟随先行者时产生herding
- Bikhchandani, Hirshleifer & Welch (1992) 进一步证明信息级联在序贯决策中不可避免
- Kukacka & Barunik (2014, Physica A) 在HAM框架中实证研究herding、overconfidence和sentiment对市场动态的影响
- Pitre et al. (2025) 的CONSENSAGENT将sycophancy（LLM特有的谄媚倾向）识别为multi-agent debate中herding的算法机制

**Anchoring的理论根基:**
- Tversky & Kahneman (1974) 奠基之作: 人类判断被初始锚定值系统性偏移
- 在multi-agent debate中，Round 1自身的预测就是一个锚点; 如果agent对新信息反应不足，就表现为anchoring
- Hashimoto et al. (2025, PRIMA) 在LLM-agent金融模拟中发现LLM agent表现出锚定于参考点的loss aversion

**Overconfidence的理论根基:**
- Moore & Healy (2008) 将overconfidence分解为overestimation、overplacement和overprecision
- 在我们的框架中，overconfidence表现为: agent在看到conflicting evidence后仍保持高置信度并做出大幅调整（overprecision）

---

## Step 7: Shapley归因 — 合作博弈论框架

### 设计
将7个agent的预测聚合视为合作博弈 $(N, v)$:
- $N = \{1, 2, \ldots, 7\}$ 为agent集合
- $v(S)$ 为子集 $S$ 的预测误差: $v(S) = (y_t - \hat{\sigma}_S)^2$，其中 $\hat{\sigma}_S$ 为仅使用子集 $S$ 中agent的预测

Agent $i$ 的Shapley值:

$$\phi_i = \frac{1}{|N|!} \sum_{\pi \in \Pi(N)} [v(S_i^\pi \cup \{i\}) - v(S_i^\pi)]$$

Monte Carlo近似: 随机采样500个排列，计算每个agent加入时的边际贡献。

$\phi_i < 0$ 表示agent $i$ 改善了预测（减少误差），$\phi_i > 0$ 表示agent $i$ 恶化了预测。

### 文献支撑

**Shapley值的理论性质:**
- Shapley (1953) 证明满足效率性、对称性、虚拟参与者和可加性四个公理的唯一解
- 在预测归因语境中，Shapley值将集体预测误差公平地分配到每个agent，同时考虑了agent之间的交互效应

**Shapley值在multi-agent系统中的应用:**
- Xia et al. (2025) 的HiveMind使用Shapley值量化multi-agent LLM系统中每个agent的贡献，提出DAG-Shapley提高计算效率。在multi-agent stock trading中验证了有效性
- Hua et al. (2025) 的Shapley-Coop用Shapley值的边际贡献作为pricing basis，解决self-interested LLM agent的协调问题
- Wang (2024) 定义"Markov Shapley value"，将Shapley值推广到序贯决策过程，证明了效率性、虚拟agent识别、贡献反映和对称性

**Shapley值在预测聚合中的应用:**
- Bimonte et al. (2024, Decisions in Economics and Finance) 用Shapley值为mortality prediction的ensemble分配权重，证明优于ad hoc加权
- 在ML可解释性领域，SHAP (Lundberg & Lee, 2017) 基于Shapley值，已成为feature attribution的金标准

**Monte Carlo近似的合理性:**
- 7个agent的精确Shapley计算需要 $2^7 = 128$ 个子集评估。我们使用500次排列采样的Monte Carlo近似
- 这一近似方法在理论上由 Castro et al. (2009) 证明收敛，并被广泛采用

---

## Step 8: Baseline体系 — 从经典到前沿的对比

### 设计
12个baseline方法分四层:

| 层级 | 方法 | 文献来源 |
|------|------|---------|
| **持续性** | Persistence ($\hat{\sigma} = \sigma_t$) | Random walk benchmark |
| **经典计量** | HAR (Corsi 2009) | 日/周/月三尺度线性模型 |
| | GARCH(1,1) (Bollerslev 1986) | 条件异方差经典模型 |
| **机器学习** | Ridge, Lasso, GBR, Random Forest | Christensen et al. (2023, JFE) |
| | XGBoost | Tiwari et al. (2024, Energy Economics) |
| **深度学习** | LSTM | Ben Romdhane & Boubaker (2026) |
| | Transformer | Qiu et al. (2025) |
| **LLM baseline** | Single-Agent (同信息，无辩论) | 消融实验: 隔离辩论机制的增量价值 |

### 文献支撑

**HAR作为核心benchmark:**
- Corsi (2009) 是已实现波动率预测领域被引用最多的模型（1000+引用），几乎所有后续工作都以HAR为baseline
- Christensen et al. (2023, JFE) 确立了ML vs HAR的标准评估协议

**GARCH的必要性:**
- Herrera et al. (2018, IJF) 在原油波动率预测中系统比较了GARCH族模型，发现GARCH(1,1)在短horizon上具有竞争力
- Zhang et al. (2019, IREF) 比较了单regime与regime-switching GARCH

**ML/DL方法的选择:**
- Christensen et al. (2023) 发现regularization和tree-based方法在RV预测中优于神经网络（以较少调参成本获得更好效果）
- Omer et al. (2026, J. of Forecasting) 直接在WTI原油上验证了ML方法相对HAR的优势
- Ben Romdhane & Boubaker (2026) 的HAR-LSTM-GARCH混合模型代表当前技术前沿

**Single-Agent消融的方法论意义:**
- Single-Agent baseline使用同一个LLM、同样的完整数据，但不经过辩论。这严格隔离了辩论机制的增量价值
- 对应Du et al. (2023) 中的"single-agent self-refinement" baseline
- Denisov-Blanch et al. (2026) 发现单纯增加LLM采样次数并不总能提升精度（共享偏差问题），因此debate的价值需要通过vs Single-Agent来论证

### 评估指标

- **RMSE**: 标准预测精度指标
- **Diebold-Mariano检验 (Diebold & Mariano, 1995)**: 成对预测能力对比的统计检验，Newey-West HAC标准误（lag=20，对应20日滚动窗口的重叠）
- **分Regime评估**: 按波动率水平分层（Low/Normal/Elevated/Crisis），对应Herrera et al. (2018) 的发现: 模型的相对优势随市场状态变化

---

## Step 9: Regime-Dependent Analysis — 行为偏差的状态依赖性

### 设计
四个波动率状态:
- Low: vol < 20%
- Normal: 20% ≤ vol < 35%
- Elevated: 35% ≤ vol < 55%
- Crisis: vol ≥ 55%

分析: 行为偏差（herding、anchoring）的频率和强度是否随regime变化？

### 文献支撑

**Regime-dependent行为的理论预期:**
- Banerjee (1992) 的信息级联理论预测: 不确定性越高，agent越倾向于放弃私有信息、跟随他人（herding增加）
- Surowiecki (2004) 的集体智慧理论预测: 高不确定性下独立性的丧失会削弱集体预测质量
- Kukacka & Barunik (2014) 在HAM框架中实证证明: overconfidence在危机期放大波动率，herding在趋势市中自我强化

**金融市场的regime-dependent特征:**
- Herrera et al. (2018) 发现不同GARCH模型在不同horizon和市场状态下表现不同
- Zhang & Zhang (2023) 比较smooth structural shifts vs sharp regime switching，证明regime划分对预测至关重要
- Bouri et al. (2020) 证明传染病不确定性对波动率的预测力在不同regime下异质性显著

---

## 方法论体系总结

```
Step 1: Realized Volatility Target
  ↓  [Corsi 2009; Andersen & Bollerslev 1998]
Step 2: Information Decomposition → 7 Specialist Agents
  ↓  [Kilian 2009; Hommes 2006; Caldara & Iacoviello 2022]
Step 3: Structured Delphi Debate (2 Rounds)
  ↓  [Dalkey & Helmer 1963; Lorenz & Fritz 2026; Du et al. 2023]
Step 4: Persistence + Adjustment Architecture
  ↓  [Mandelbrot 1963; Engle 1982; Halawi et al. 2024]
Step 5: Confidence-Weighted Aggregation
  ↓  [Bates & Granger 1969; Hanea et al. 2021; Chen et al. 2024]
Step 6: Behavioral Bias Classification
  ↓  [Banerjee 1992; Ashery et al. 2025; Pitre et al. 2025]
Step 7: Shapley Value Attribution
  ↓  [Shapley 1953; Xia et al. 2025; Lundberg & Lee 2017]
Step 8: Multi-Layer Baseline Comparison
  ↓  [Christensen et al. 2023; Herrera et al. 2018; Diebold & Mariano 1995]
Step 9: Regime-Dependent Analysis
       [Banerjee 1992; Kukacka & Barunik 2014; Bouri et al. 2020]
```

## 核心参考文献

### 已实现波动率预测
- Andersen, T.G. & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*, 39(4), 885-905.
- Andersen, T.G., Bollerslev, T. & Diebold, F.X. (2007). Roughing it up: Including jump components. *Review of Economics and Statistics*, 89(4), 701-720.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Christensen, K., Siggaard, M. & Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics*, 21(5), 1680-1727.

### 原油波动率
- Herrera, A.M., Hu, L. & Pastor, D.J. (2018). Forecasting crude oil price volatility. *International Journal of Forecasting*, 34(4), 622-635.
- Bouri, E., Demirer, R., Gupta, R. & Pierdzioch, C. (2020). Infectious diseases, market uncertainty and oil market volatility. *Energies*, 13(16), 4090.
- Nonejad, N. (2021). Forecasting crude oil price volatility out-of-sample using news-based geopolitical risk index. *Finance Research Letters*, 44, 102310.
- Kilian, L. (2009). Not all oil price shocks are alike. *American Economic Review*, 99(3), 1053-1069.

### Multi-Agent LLM Debate
- Du, Y., Li, S., Torralba, A., Tenenbaum, J.B. & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.
- Liang, T. et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *EMNLP 2024*.
- Chen, J.C.-Y., Saha, S. & Bansal, M. (2024). ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. *ACL 2024*.
- Pitre, P., Ramakrishnan, N. & Wang, X. (2025). CONSENSAGENT: Towards efficient consensus in multi-agent LLM interactions through sycophancy mitigation. *Findings of ACL*.

### LLM-Delphi
- Dalkey, N. & Helmer, O. (1963). An experimental application of the Delphi method to the use of experts. *Management Science*, 9(3), 458-467.
- Lorenz, T. & Fritz, M. (2026). Scalable Delphi: Large language models for structured risk estimation. *arXiv:2602.08889*.
- Schoenegger, P. et al. (2024). Wisdom of the silicon crowd: LLM ensemble prediction capabilities rival human crowd accuracy. *Science Advances*.

### Forecast Combination
- Bates, J.M. & Granger, C.W.J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451-468.
- Wang, X., Hyndman, R.J., Li, F. & Kang, Y. (2023). Forecast combinations: An over 50-year review. *International Journal of Forecasting*, 39(4), 1518-1547.
- Hanea, A.M. et al. (2021). Mathematically aggregating experts' predictions of possible futures. *PLoS ONE*.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253-263.

### 行为偏差
- Banerjee, A.V. (1992). A simple model of herd behavior. *Quarterly Journal of Economics*, 107(3), 797-817.
- Bikhchandani, S., Hirshleifer, D. & Welch, I. (1992). A theory of fads, fashion, custom, and cultural change as informational cascades. *Journal of Political Economy*, 100(5), 992-1026.
- Tversky, A. & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.
- Ashery, A.F., Aiello, L.M. & Baronchelli, A. (2025). Emergent social conventions and collective bias in LLM populations. *Science Advances*.
- Kukacka, J. & Barunik, J. (2014). Behavioural breaks in the heterogeneous agent model. *Physica A*, 392(23), 5920-5938.
- Hommes, C.H. (2006). Heterogeneous agent models in economics and finance. *Handbook of Computational Economics*, Vol. 2, 1109-1186.

### Shapley归因
- Shapley, L.S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2, 307-317.
- Xia, Y. et al. (2025). HiveMind: Contribution-guided online prompt optimization of LLM multi-agent systems. *arXiv:2512.06432*.
- Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

### 异质Agent模型
- Axtell, R.L. & Farmer, J.D. (2025). Agent-based modeling in economics and finance: Past, present, and future. *Journal of Economic Literature*, 63(1), 197-287.
- Caldara, D. & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194-1225.
- Diebold, F.X. & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. *International Journal of Forecasting*, 28(1), 57-66.

### LLM预测能力
- Halawi, D., Zhang, F., Chen, Y.-H. & Steinhardt, J. (2024). Approaching human-level forecasting with language models. *NeurIPS 2024*.
- Tan, M. et al. (2024). Are language models actually useful for time series forecasting? *NeurIPS 2024*.
- Williams, A.R. et al. (2024). Context is key: A benchmark for forecasting with essential textual information. *arXiv:2410.18959*.
- Denisov-Blanch, Y. et al. (2026). Consensus is not verification: Why crowd wisdom strategies fail for LLM truthfulness. *arXiv:2603.06612*.
