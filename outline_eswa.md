# 论文大纲 (ESWA版)

## Title
Causal Graph Attention over Multi-Agent LLM Debate for Interpretable Oil Volatility Forecasting

## Abstract要点
- 油价波动率预测对能源风险管理至关重要，但现有方法要么精度高不可解释（深度学习堆叠），要么可解释但精度有限（HAR/GARCH）
- 提出两阶段框架：(1) 7个专家LLM agent通过结构化Delphi辩论生成异质预测，(2) Causal-GAT根据agent因果交互图自适应聚合
- 1,285个交易日(2020-2025)的walk-forward评估，涵盖COVID负油价、俄乌冲突、中东升级等极端事件
- Causal-GAT RMSE=0.1368，显著优于HAR (DM=-9.83)，同时通过Shapley归因提供driver-level可解释性

---

## 1. Introduction

1. 油价波动率预测在能源风险管理、对冲决策、期权定价中的核心地位
2. 现有方法的两难：深度学习精度高但黑箱 vs 计量经济模型可解释但精度有限
   - HAR (Corsi 2009) 仍是标准benchmark
   - 近年CEEMDAN+LSTM等混合模型追求精度但不可解释 (Tiwari et al. 2024)
   - LLM目前仅被用作情绪提取工具 (Dai et al. 2026)，未作为推理agent
3. 多agent LLM辩论在金融领域的兴起 (TradingAgents/ICML 2025; Takano et al. 2025)，但存在两个未解决问题：
   - Agent间herding导致集体偏差，naive聚合无法处理
   - 缺乏agent级别的可解释归因
4. 本文贡献（三点）：
   - C1: 首个将multi-agent LLM debate应用于商品波动率预测的框架（Structured Delphi Debate），7个agent对应油市分析的真实专业分工
   - C2: Causal-GAT聚合机制，通过Granger因果发现识别agent间真实信息传递 vs herding，用图注意力网络学习regime-adaptive聚合权重
   - C3: Agent级Shapley归因，将预测分解为各信息源的贡献，提供driver-level可解释性

## 2. Related Work

### 2.1 Oil Volatility Forecasting
- 计量经济方法：GARCH族 (Bollerslev 1986)、HAR-RV (Corsi 2009)
- ML方法：Ridge/Lasso/GBR/RF、XGBoost
- DL方法：LSTM、Transformer、混合分解模型
- Automated ML: Li & Tang (2024, MS) 118特征+5算法集成做股票波动率
- Gap: 精度竞赛与可解释性的矛盾

### 2.2 LLM for Financial Forecasting
- LLM as sentiment extractor: FinGPT (Liu et al. 2023), Dai et al. (2026) WTI多维情绪
- LLM as reasoning agent: TradingAgents (Xiao et al. 2024), FinCon (Yu et al. 2024)
- Multi-agent debate: Du et al. (2024) factuality improvement, Takano et al. (2025) FOMC预测
- Gap: 无人将multi-agent LLM用于连续值波动率预测；无因果感知的聚合机制

### 2.3 Emergent Bias in Multi-Agent LLM Systems
- Herding/groupthink: Ashery et al. (2024) Science Advances, Wang et al. (2025) Catfish
- Overconfidence amplification: Sun et al. (2025), Bini et al. (2026)
- Gap: 已识别偏差但未提出利用偏差诊断来改善聚合的方法

### 2.4 Explainable AI for Forecasting
- SHAP for feature importance (主流XAI工具)
- Shapley value in MARL credit assignment (Wang et al. 2020, 2022)
- Causal Shapley: Heskes et al. (2020)
- Gap: Shapley用于agent级归因（而非feature级）的工作极少

## 3. Methodology

### 3.1 Problem Formulation
- 预测目标：前瞻性20日已实现波动率 fwd_rv_20d
- 公式定义
- Walk-forward evaluation设定

### 3.2 Stage 1: Multi-Agent Structured Delphi Debate

#### 3.2.1 Agent Design
- 7个specialist agents及其信息域：
  - Geopolitical: 地缘政治风险事件 (GDELT数据)
  - Macro Demand: 宏观经济需求信号 (VIX, 上证综指)
  - Monetary: 货币政策环境 (联邦基金利率, 收益率曲线)
  - Supply/OPEC: 供给端动态
  - Technical: 价格技术指标和动量
  - Sentiment: 市场情绪和波动率结构
  - Cross-Market: 跨市场联动 (DXY, 利差)
- 每个agent输出: adjustment (相对于persistence的调整量)、confidence、evidence、direction

#### 3.2.2 Debate Protocol
- Round 1: 独立预测（保证信息多样性）
- Round 2: 看到所有agent的Round 1预测和理由后修正
- 结构化JSON输出格式

#### 3.2.3 Influence Graph Construction
- 基于Round 1→Round 2的adjustment变化检测influence edges
- 边权重定义

#### 3.2.4 Behavioral Classification
- Herding: 向群体均值收敛
- Anchoring: 几乎不调整
- Independent: 基于新信息独立调整
- Overconfident: 逆群体方向强化

### 3.3 Stage 2: Causal-GAT Aggregation

#### 3.3.1 Causal Discovery
- Granger因果检验 on agent Shapley值时间序列
- 因果邻接矩阵构建
- 区分genuine information transfer vs spurious herding correlation

#### 3.3.2 Graph Attention Network
- Node features: Shapley, Myerson, behavior, degree, rolling statistics
- Context features: base predictions, regime, herding level
- 2层GAT architecture
- Softmax weights over 5 base forecasts + residual correction

#### 3.3.3 Training Protocol
- Walk-forward: min_train=252, retrain_every=63
- Cosine LR schedule, best checkpoint selection

### 3.4 Attribution Framework

#### 3.4.1 Shapley Value for Agent Attribution
- Agent作为cooperative game的player
- Monte Carlo近似
- 反事实基线：agent替换为零调整

#### 3.4.2 Myerson Value for Path Attribution
- 结合influence graph的图限制Shapley
- 识别influence pathway

## 4. Experimental Setup

### 4.1 Data
- WTI原油价格 + 宏观指标 (2011-2025)
- GDELT地缘政治事件特征
- 评估期: 2020-01 ~ 2025-05 (1,285交易日)
- 负油价处理: 排除actual_vol > 2的极端天(61天), clean sample n=1,224

### 4.2 Baselines
- Naive: Persistence, Historical Mean
- Econometric: GARCH, HAR
- ML: Ridge, Lasso, GBR, RF
- DL: XGBoost, LSTM, Transformer
- LLM: Single-agent (无辩论)
- Ablation: Naive aggregation (confidence-weighted mean), Simple blend

### 4.3 Evaluation Metrics
- RMSE, MAE
- Diebold-Mariano test
- Regime-conditional RMSE (low/normal/elevated/high volatility)

### 4.4 Implementation
- LLM: Gemini 3 Flash
- Causal-GAT: PyTorch, 2层GAT, hidden_dim=16

## 5. Results

### 5.1 Overall Prediction Performance
- 全模型RMSE排名表
- DM检验: Causal-GAT vs 各baseline
- 关键结果: Causal-GAT (0.1368) 显著优于HAR (0.1549, DM=-9.83), Debate naive (0.1512), 所有ML/DL baselines

### 5.2 Regime-Conditional Analysis
- 分regime RMSE对比
- Causal-GAT在所有regime均优于HAR
- 高波动regime改善最大 (0.155 vs 0.223, 降幅30.6%)

### 5.3 Causal-GAT Weight Analysis
- Learned weights的regime依赖性
- GARCH被完全抛弃 (0.2%), Single agent获最高权重 (43%)
- 高波动时HAR权重自动升至46%
- 解释: 模型学会了regime-adaptive的信任分配

### 5.4 Agent Attribution Analysis
- 7个agent的Shapley值分布
- 所有agent Shapley均为负（均改善预测）
- 不同regime下的dominant agent变化
- Case studies: COVID crash, Russia-Ukraine, 中东升级期的归因分解

### 5.5 Herding Dynamics
- Herding占比42.9%，technical和cross_market最高(52%)
- Herding与通信密度强相关 (r=0.37)
- Herding与预测误差的关系: 控制regime后不显著 → herding不全是坏事
- Causal-GAT如何应对herding: 自动shift weights toward HAR

### 5.6 Ablation Studies
- Without causal discovery (random graph) vs with causal graph
- Without GAT (simple linear blend) vs with GAT
- Without debate (pure statistical) vs with debate information
- Stage 1 only (naive aggregation) vs Stage 1 + Stage 2 (Causal-GAT)

## 6. Discussion

### 6.1 Why Multi-Agent Debate Adds Value
- All agent Shapley values negative: each information source improves forecast
- The bottleneck was aggregation, not information generation
- Causal-GAT解决了aggregation bottleneck

### 6.2 Interpretability for Risk Management
- Agent归因 → 风险经理知道什么因素在驱动波动率变化
- Regime-adaptive weights → 系统自动告知"何时信什么"
- Herding检测 → 预测共识度/不确定性的proxy

### 6.3 Comparison with Existing Approaches
- vs Li & Tang (2024 MS): 我们做commodity不做equity，且提供agent级归因
- vs Dai et al. (2026): LLM从被动情绪提取升级为主动推理agent
- vs TradingAgents: 从交易决策扩展到波动率预测，加因果聚合

### 6.4 Limitations
- 单一LLM (Gemini 3 Flash) 依赖性
- Causal-GAT的walk-forward评估丢失前252天
- 7个agent的角色设计基于领域知识，非自动发现
- 负油价等极端事件样本量有限

### 6.5 Future Work
- 多LLM ensemble (不同model扮演不同agent)
- 动态agent数量/角色的自适应调整
- 扩展到其他大宗商品 (天然气、金属)
- 在线学习: Causal-GAT实时更新

## 7. Conclusion
- 总结两阶段框架和三个贡献
- 强调精度+可解释性的双重价值
- 对能源风险管理实践的意义
