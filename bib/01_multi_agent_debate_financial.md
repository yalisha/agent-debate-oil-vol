# Multi-Agent LLM Debate Systems for Financial Forecasting

Literature survey conducted 2026-03-18.

---

## 1. Foundational Work

### Du et al. (2024) - Improving Factuality and Reasoning through Multiagent Debate
- **Venue:** ICML 2024
- **arXiv:** 2305.14325
- **Method:** Multiple LLM instances propose and debate responses over multiple rounds to converge on a common answer.
- **Findings:** Significantly enhances mathematical and strategic reasoning, reduces hallucinations.
- **Relevance:** Foundational methodology that most financial debate papers build upon.

---

## 2. Multi-Agent Debate in Finance

### FactorMAD (Duan et al., 2025) - Multi-Agent Debate for Alpha Factor Mining
- **Venue:** ACM ICAIF 2025
- **Method:** Two GPT-4o agents engage in structured multi-round debate, alternating between proposing alpha factors and critiquing them. Code-based factor development pipeline.
- **Findings:** Outperforms automated factor mining methods in predictive performance and trading profitability.
- **Interpretability:** Factors expressed as readable code with natural language rationales.
- **Key takeaway:** Debate mechanism enables discovery of more effective predictive signals through diverse perspectives.

### TradingAgents (Xiao et al., 2024) - Multi-Agents LLM Financial Trading
- **arXiv:** 2412.20138
- **Venue:** ICML 2025 oral
- **Method:** Seven agent roles (Fundamentals/Sentiment/News/Technical Analyst, Researcher, Trader, Risk Manager). Bull and Bear researchers engage in structured debate. Uses web search, Reddit/Twitter APIs, sentiment scoring.
- **Findings:** Improved cumulative returns, Sharpe ratio, and max drawdown vs baselines.
- **Interpretability:** Natural language operations ensure transparency; debate transcripts are traceable.

### Hawkish-Dovish Latent Beliefs (Takano et al., 2025) - Multi-Agent Debate for Monetary Policy
- **arXiv:** 2511.02469
- **Venue:** PRIMA 2025
- **Method:** LLM agents with distinct belief profiles (Strong Hawkish to Strong Dovish) iteratively revise predictions by observing others' outputs, simulating FOMC deliberation.
- **Findings:** Significantly outperforms standard LLM baselines in monetary policy prediction.
- **Key takeaway:** Explicit modeling of heterogeneous beliefs + iterative revision is a powerful pattern. Most similar to our proposed debate mechanism.

### TradingGPT (Li et al., 2023) - Layered Memory and Distinct Characters
- **arXiv:** 2309.03736
- **Method:** LLM agents with hierarchical memory and distinct traits. Inter-agent debate for overlapping positions using top-K memories and recommendations.
- **Findings:** Debate improves robustness vs isolated single-agent systems.
- **Key takeaway:** Layered memory provides audit trails; character differentiation matters.

### HAD (Xing, 2024) - Heterogeneous Agent Discussion for Financial Sentiment
- **arXiv:** 2401.05799
- **Venue:** ACM TMIS
- **Method:** 7 specialized agents (mood, rhetoric, dependency, etc.) inspired by Minsky's theory of mind. Shared discussion mechanism.
- **Findings:** Better than homogeneous debate; fixes 20-50% of the gap between ICL and fine-tuning.
- **Key takeaway:** Heterogeneous agent design outperforms homogeneous debate.

### FOMC In Silico (Kazinnik & Sinclair, 2025) - Simulating FOMC Debates
- **Venue:** GWU Working Paper
- **Method:** LLM agents modeled after real FOMC members debate and vote on monetary policy. Dual-track: LLM debate + Monte Carlo Bayesian voting benchmark.
- **Findings:** Predicted 4.42% (LLM) and 4.38% (MC), near actual range. 88% dissent under political pressure.
- **Key takeaway:** Agent-level reasoning visible through debate transcripts; dissent patterns provide attribution.

---

## 3. Multi-Agent Financial Systems (Non-Debate)

### FinCon (Yu et al., 2024) - Conceptual Verbal Reinforcement
- **arXiv:** 2407.06567
- **Venue:** NeurIPS 2024
- **Method:** Manager-analyst hierarchy. Analysts process single-source info; risk-control triggers self-critiquing to update beliefs.
- **Findings:** Improved returns while reducing communication costs.

### P1GPT (Lu et al., 2025) - Multi-Modal Financial Analysis
- **arXiv:** 2510.23032
- **Method:** Four-layer architecture (Input, Planning, Analysis, Integration). Domain-specific specialist agents for fundamental/technical/news/sectoral analysis.
- **Findings:** Superior risk-adjusted returns with transparent causal rationales.

### QuantAgents (Li et al., 2025) - Simulated Trading Meetings
- **arXiv:** 2510.04643
- **Venue:** EMNLP 2025 Findings
- **Method:** Four agents collaborate via structured meetings. Feedback from both real market and simulated trading.
- **Findings:** ~300% return over three years.

### QuantAgent (Xiong et al., 2025) - Price-Driven HFT
- **arXiv:** 2509.09995
- **Method:** Four specialized agents (Indicator, Pattern, Trend, Risk) operating on price-derived signals only.
- **Findings:** Up to 80% directional accuracy at 1h and 4h intervals across 9 instruments.

### Ploutos (Tong et al., 2024) - Interpretable Stock Movement Prediction
- **arXiv:** 2403.00782
- **Venue:** WWW 2025
- **Method:** Multiple primary experts + PloutosGPT combining insights with interpretable rationales. Rearview-mirror prompting.
- **Findings:** SOTA on prediction accuracy and interpretability.

### MASFIN (Montalvo et al., 2025) - Decomposed Financial Reasoning
- **arXiv:** 2512.21878
- **Method:** Five crews of 3-5 LLM agents. Postmortem Crew identifies risks, failure patterns.
- **Findings:** 7.33% cumulative return over 8 weeks, outperforming major indices.

### FinVision (Fatemi & Hu, 2024) - Stock Market Prediction with Reflection
- **arXiv:** 2411.08899
- **Venue:** ACM ICAIF 2024
- **Method:** Four modules including Reflection Module analyzing historical trading outcomes.
- **Findings:** Outperforms market benchmark on AAPL and MSFT.

---

## 4. Oil/Commodity Specific

### Beyond Polarity (Dai et al., 2026) - Multi-Dimensional LLM Sentiment for WTI
- **arXiv:** 2603.11408
- **Method:** Five sentiment dimensions from energy-sector news using GPT-4o, Llama, FinBERT. SHAP analysis.
- **Findings:** Combining GPT-4o and FinBERT yields best results. Intensity and uncertainty features most important.
- **Relevance:** Directly about WTI crude oil; uses SHAP for feature attribution.

### Commodity Price Shocks with Agentic AI (Ghali et al., 2025)
- **arXiv:** 2508.06497
- **Method:** Agentic generative AI pipeline + dual-stream LSTM fusing price time-series with semantically embedded news (1960-2023).
- **Findings:** AUC 0.94, accuracy 0.91. Removing news drops AUC to 0.46.
- **Relevance:** Commodity shock prediction using agentic AI pipeline.

---

## 5. Argumentation-Based Forecasting

### Retrieval- and Argumentation-Enhanced Multi-Agent Forecasting (Gorur et al., 2025)
- **arXiv:** 2510.24303
- **Venue:** AAMAS 2026
- **Method:** Three agent types using quantitative bipolar argumentation frameworks (QBAFs). Forecasting as claim verification.
- **Findings:** Combining evidence from three agents improves accuracy with explainable evidence.
- **Key takeaway:** Formal argumentation structure for prediction is a viable approach.

---

## 6. Graph-Structured Agent Communication

### G-Designer (Zhang et al., 2024) - Communication Topologies via GNN
- **arXiv:** 2410.11782
- **Method:** Variational graph auto-encoder encodes agents + task node, decodes task-adaptive communication topology.
- **Findings:** 84.50% MMLU, 89.90% HumanEval. Up to 95.33% token reduction.
- **Key takeaway:** Communication topology matters; dynamic graph design outperforms fixed topologies.

### GTD (Jiang et al., 2025) - Graph Diffusion for Communication Topologies
- **arXiv:** 2510.07799
- **Method:** Conditional discrete graph diffusion models for iterative topology construction.
- **Findings:** Task-adaptive, sparse, efficient topologies balancing accuracy, cost, resilience.

### AMAS (2025) - Adaptively Determining Communication Topology
- **arXiv:** 2510.01617
- **Venue:** EMNLP 2025 Industry Track
- **Method:** Lightweight LLM adaptation to identify task-specific optimal graph configurations.
- **Findings:** Systematically exceeds baselines across QA, math, and code generation.

### Graph Attention-Based Multi-Agent Portfolio (2025)
- **Venue:** Scientific Reports (Nature)
- **Method:** Graph attention networks model time-varying asset correlations. Three heterogeneous agents (risk, return, environment).
- **Findings:** 16.8% annualized returns, 1.34 Sharpe, 8.2% max drawdown.
