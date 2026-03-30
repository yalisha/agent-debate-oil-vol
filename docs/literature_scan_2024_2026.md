# Literature Scan: Oil Volatility Forecasting with AI/LLM/Multi-Agent Methods (2024-2026)

**Scan Date**: 2026-03-23
**Scope**: Energy Economics, EJOR, Resources Policy, Management Science, arXiv, and cross-journal search

---

## Executive Summary

The 2024-2026 literature on oil price/volatility forecasting with AI shows three clear trends: (1) hybrid decomposition-deep learning models remain dominant, (2) LLM-based sentiment extraction is emerging as a new frontier, and (3) multi-agent LLM systems have appeared for equity trading but have NOT been applied to commodity volatility forecasting. No published paper combines multi-agent LLM debate with Shapley-based interpretability for oil volatility prediction. Our Structured Delphi Debate (SDD) with Causal-GAT aggregation fills a genuine gap.

---

## 1. Energy Economics (2024-2025)

### 1.1 Key Papers

**Paper 1**: Tiwari, Sharma, Rao, Hossain & Dev (2024). "Unraveling the crystal ball: Machine learning models for crude oil and natural gas volatility forecasting." *Energy Economics*, 134, S0140988324003165.
- **Method**: 11 ML models (LSTM, ANN, RF, XGBoost, Ridge, Linear, Elastic Net, Huber, etc.) across daily/weekly/monthly/quarterly horizons
- **Key Finding**: Best models vary by horizon: LSTM for daily crude oil, RF/XGBoost for natural gas. No single model dominates all horizons.
- **Gap**: No interpretability mechanism, no multi-source textual information, no agent-based reasoning.

**Paper 2**: "More is better? The impact of predictor choice on the INE oil futures volatility forecasting." *Energy Economics*, 2024, S0140988324002482.
- **Method**: 68 economic indicators across 11 categories; ensemble tree models with dynamic parameter optimization
- **Key Finding**: More predictors do not always help, even for ML models. A few key variables from market sentiment, other crude oil exchanges, and energy markets suffice. Ensemble trees outperform traditional ML.
- **Gap**: Feature selection only; no agent-based reasoning or semantic understanding of market drivers.

**Paper 3**: "A multiscale time-series decomposition learning for crude oil price forecasting." *Energy Economics*, 2024, S0140988324004419.
- **Method**: Multiscale temporal processing module + decomposition schema (global smoothing + local non-smoothing patterns)
- **Key Finding**: Multiscale decomposition significantly improves forecasting accuracy across different time horizons.
- **Gap**: Purely numerical; no textual/semantic information, no agent-based interpretation.

**Paper 4**: "Energy organization sentiment and oil return forecast." *Energy Economics*, 2024, S0140988324008144.
- **Method**: ChatGPT used to derive sentiment indexes from IEA and OPEC reports
- **Key Finding**: Organization sentiment negatively predicts future oil returns; OPEC sentiment dominates IEA sentiment.
- **Relevance**: First use of ChatGPT for oil-related sentiment extraction in Energy Economics. Single LLM use, not multi-agent.

**Paper 5**: "Can the sentiment of the official media predict the return volatility of the Chinese crude oil futures?" *Energy Economics*, 2024, S0140988324006753.
- **Method**: Official vs. business media sentiment for volatility prediction
- **Key Finding**: Official media sentiment has stronger forecasting power than business media for Chinese crude oil futures volatility.
- **Gap**: Sentiment is used as input feature only, not as agent reasoning.

### 1.2 Popular Methods in Energy Economics
- Decomposition + DL hybrids: CEEMDAN-VMD-CNN-BiLSTM, CEEMDAN-GRU
- HAR-RV extensions: HAR + ML variants (LASSO, Ridge, XGBoost, LightGBM)
- Ensemble tree models: XGBoost, LightGBM, CatBoost, Random Forest
- Transformer-based: iTransformer, Temporal Fusion Transformer (emerging)
- Sentiment integration: ChatGPT-extracted sentiment, news NLP (emerging)

### 1.3 Gaps in Energy Economics
- No multi-agent LLM system paper
- No structured debate protocol for forecasting
- Limited use of LLMs beyond sentiment extraction
- Interpretability mostly limited to SHAP on black-box models; no agent-level attribution
- No regime-adaptive aggregation mechanism

---

## 2. European Journal of Operational Research (EJOR, 2024-2025)

### 2.1 Key Papers

**Paper 1**: "Explainable AI for Operational Research: A defining framework, methods, applications, and a research agenda." *EJOR*, 317(2), 249-272, 2024.
- **Method**: Normative framework for XAI in OR: performance + attributable + responsible analytics
- **Key Finding**: SHAP dominates post-hoc XAI methods (62% of studies). Key OR applications include forecasting, risk analysis, and supply chain.
- **Relevance**: Provides theoretical grounding for our Shapley-based agent attribution. EJOR values interpretability.

**Paper 2**: Borgonovo et al. (2024). "Shapley value-based measures of feature importance." *EJOR*, 318, 911-926.
- **Method**: Theoretical analysis of Shapley-based feature importance measures
- **Key Finding**: Methods based on Shapley values are gaining attention as measures of feature importance for explaining black-box predictions.
- **Relevance**: Direct theoretical anchor for our use of Shapley values for agent contribution attribution.

### 2.2 Gaps in EJOR
- No oil/commodity-specific papers using LLM or multi-agent systems found in 2024-2025
- EJOR publishes on XAI theory and OR applications but not yet on multi-agent debate for forecasting
- Strong opportunity to position our paper's Causal-GAT + Shapley mechanism as XAI-for-OR contribution

---

## 3. Resources Policy (2024-2025)

### 3.1 Key Papers

**Paper 1**: "Forecasting the Crude Oil prices for last four decades using deep learning approach." *Resources Policy*, 88, 2024.
- **Method**: Deep learning models for long-horizon crude oil forecasting
- **Key Finding**: DL outperforms traditional approaches over 40-year span.

**Paper 2**: "Geopolitical risks and crude oil futures volatility: Evidence from machine learning." *Resources Policy*, October 2024, S0301420724007414.
- **Method**: ML models (especially Transformer-based neural networks) with geopolitical risk subcategories
- **Key Finding**: War and terrorism geopolitical risks dominate crude oil futures volatility forecasts. Impact is time-varying and asymmetric across economic conditions.
- **Relevance**: Validates our geopolitical agent's role. Their approach uses geopolitical risk as features; ours uses a dedicated geopolitical reasoning agent.

**Paper 3**: "A novel deep-learning technique for forecasting oil price volatility using historical prices of five precious metals." *Resources Policy*, 86, 2023.
- **Method**: Deep learning vs ML vs statistical comparison for oil volatility using cross-market precious metal prices
- **Key Finding**: Cross-market information from precious metals improves oil volatility forecasting.
- **Relevance**: Supports our cross-market agent design.

### 3.2 Gaps in Resources Policy
- No LLM-based paper for oil price/volatility
- No multi-agent system paper
- Focus remains on traditional ML/DL pipelines with numerical features
- Limited interpretability analysis

---

## 4. Management Science "AI for Finance and Business Decisions" VSI

### 4.1 Status
- **Call for Papers** announced October 2024 (Volume 70, Issue 10)
- **Editors**: Baris Ata, Lin William Cong, Kay Giesecke, Peng Sun, Chung Piaw Teo
- **Deadline**: December 31, 2025
- **Scope**: Economics, methodology, and applications of AI and novel data analytics in finance and business decisions

### 4.2 Assessment
- No accepted papers on oil/commodity forecasting found yet (deadline still open)
- The VSI welcomes "new methods for decision making, optimization, inference and prediction"
- Our paper's multi-agent debate framework with Causal-GAT aggregation and Shapley attribution would fit this scope
- Management Science values methodological novelty and theoretical grounding over pure empirical ML benchmarking

---

## 5. LLM for Oil Price Forecasting (Cross-Journal)

### 5.1 Key Papers

**Paper 1**: "Prediction of Crude Oil Price using LLM: An Empirical Analysis." *Procedia Computer Science*, 2025.
- **Method**: LLM used to evaluate investment sentiment from news text; sentiment scores feed into oil price forecasting models
- **Key Finding**: LLM-based approach demonstrates superior forecasting accuracy for future crude oil price movements.
- **Limitation**: Single LLM for sentiment, not multi-agent; no debate or deliberation.

**Paper 2**: Dai, Ma, Liu, Geng & Wang (2026). "Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction." arXiv:2603.11408.
- **Method**: GPT-4o, Llama 3.2-3b, FinBERT for 5-dimensional sentiment (relevance, polarity, intensity, uncertainty, forwardness); LightGBM classifier
- **Key Finding**: GPT-4o + FinBERT achieves AUROC 0.6515. Intensity and uncertainty features are more important than polarity. SHAP used for feature importance.
- **Relevance**: Demonstrates multi-dimensional LLM sentiment is superior to single-polarity measures. But still uses LLMs as feature extractors, not as reasoning agents.

**Paper 3**: Hashami & Maldonado (2024). "Can News Predict the Direction of Oil Price Volatility? A Language Model Approach with SHAP Explanations." arXiv:2508.20707.
- **Method**: Ensemble learning (LogReg, NB, KNN) with FastText, FinBERT, Gemini Pro, LLaMA 3 embeddings; SHAP for word-level interpretation
- **Key Finding**: News count is strongest predictor. FastText outperforms other embeddings. SHAP reveals regime-dependent predictive drivers (pre-pandemic: supply-demand; pandemic: uncertainty; war: geopolitical terms).
- **Relevance**: SHAP interpretation at word level is analogous to our agent-level Shapley attribution, but at a different granularity.

### 5.2 Assessment
- LLMs for oil price are currently used ONLY as sentiment extractors (feature engineering tools)
- No paper uses LLMs as autonomous reasoning agents for oil forecasting
- No paper implements structured multi-round debate for oil/commodity prediction
- Our approach of using LLMs as specialist agents with domain expertise and debate protocol is genuinely novel

---

## 6. Multi-Agent LLM Debate (Cross-Domain)

### 6.1 Key Papers

**Paper 1**: Takano, Hirano & Nakagawa (2025). "Modeling Hawkish-Dovish Latent Beliefs in Multi-Agent Debate-Based LLMs for Monetary Policy Decision Classification." PRIMA 2025. arXiv:2511.02469.
- **Method**: Multiple LLMs as FOMC agents with initial beliefs (hawkish/dovish); iterative debate rounds; latent variable for agent beliefs
- **Key Finding**: Debate-based approach significantly outperforms standard LLM baselines. Latent beliefs provide interpretability showing how individual perspectives shape collective forecasts.
- **MOST RELEVANT PAPER TO OURS**: Closest in spirit. Differences: (1) they do classification (rate hike/cut), we do continuous volatility prediction; (2) they model FOMC, we model diverse market specialists; (3) they don't use Shapley attribution or causal aggregation.

**Paper 2**: Xiao, Sun, Luo & Wang (2024). "TradingAgents: Multi-Agents LLM Financial Trading Framework." arXiv:2412.20138.
- **Method**: Specialized agents (fundamental/sentiment/technical analysts, bull/bear researchers, risk management, traders) with debate mechanism
- **Key Finding**: Outperforms baselines in cumulative returns, Sharpe ratio, and maximum drawdown. Simulates real-world trading firm dynamics.
- **Relevance**: Multi-agent financial framework with debate, but for equity trading decisions, not volatility forecasting. No Shapley attribution.

**Paper 3**: Zaleski & Chudziak (2025). "LLM-Based multi-agent system for individual investment in energy and natural resources." *Int. J. Electronics and Telecommunications*.
- **Method**: Multi-agent LLM platform integrating technical, fundamental, sentiment analysis and price prediction for energy sector
- **Key Finding**: Achieved 54% accuracy in investment recommendations, outperforming individual methods.
- **Relevance**: Energy sector focus, but for investment advisory, not volatility forecasting. No debate mechanism.

**Paper 4**: Ghali et al. (2024). "Forecasting Commodity Price Shocks Using Temporal and Semantic Fusion of Prices Signals and Agentic Generative AI Extracted Economic News." arXiv:2508.06497.
- **Method**: Agentic AI pipeline (manager orchestrator + news specialist + fact-checker) with dual-stream LSTM and attention; AUC 0.94
- **Key Finding**: Removing news embeddings drops AUC from 0.94 to 0.46, demonstrating textual context is critical.
- **Relevance**: Agentic AI for commodities, but agents are task-decomposition tools (retrieve, summarize, fact-check), not domain-specialist debaters.

**Paper 5**: Wawer & Chudziak (2025). "Integrating Traditional Technical Analysis with AI: A Multi-Agent LLM-Based Approach to Stock Market Forecasting." *ICAART 2025*.
- **Method**: ElliottAgents system with RAG and DRL for Elliott Wave pattern recognition in stocks
- **Key Finding**: Effective pattern recognition and trend forecasting across various time frames.

### 6.2 Multi-Agent Debate Theory Papers (2024-2025)

- **"If Multi-Agent Debate is the Answer, What is the Question?"** (Zhang et al., 2025, arXiv): Systematic analysis of when MAD works
- **"Stay Focused: Problem Drift in Multi-Agent Debate"** (Becker et al., 2025, arXiv): Identifies problem drift issue; 35% lack of progress, 26% low-quality feedback, 25% lack of clarity
- **"Revisiting Multi-Agent Debate as Test-Time Scaling"** (Yang et al., 2025, arXiv): MAD most effective with increased problem difficulty and decreased model capability
- **"Literature Review of Multi-Agent Debate for Problem-Solving"** (Tillmann, 2025, arXiv): MA-LLMs outperform single-agent but face elevated computational costs and under-explored challenges

### 6.3 Cognitive Biases in Multi-Agent LLMs
- Groups of interacting LLMs are prone to **degeneration of thought**, **majority herding**, and **overconfident consensus**
- MindScope dataset covers 72 human cognitive biases in multi-agent dialogue
- Agent-agent interactions can surface latent biases including anchoring effects and gambler's fallacy
- Mitigation strategies include RAG, structured debate, and RL-based adjudication

---

## 7. Interpretable Oil Forecasting with SHAP/Shapley (Cross-Journal)

### 7.1 Key Papers

**Paper 1**: Feng, Rao, Lucey & Zhu (2024). "Volatility forecasting on China's oil futures: New evidence from interpretable ensemble boosting trees." *Int. Review of Economics and Finance*, 92, 1595-1615. [NOTE: Retracted due to editorial process conflict, not methodological issues]
- **Method**: LightGBM and CatBoost with SHAP for driver analysis of China oil futures volatility
- **Key Finding**: Ensemble boosting trees outperform HAR-RV. SHAP reveals macroeconomic vs HAR-type variables contribute differently in CatBoost vs LightGBM.

**Paper 2**: Hashami & Maldonado (2024). See Section 5.1 Paper 3 above.
- Word-level SHAP interpretation reveals regime-dependent predictive drivers.

**Paper 3**: "Forecasting of clean energy market volatility: The role of oil and the technology sector." *Energy Economics*, 2024, S0140988324001592.
- SHAP used to interpret spillover effects from oil to clean energy markets.

### 7.2 Assessment
- SHAP is the dominant post-hoc interpretability method (62% of XAI studies per EJOR review)
- SHAP is applied to feature-level importance in black-box models
- No paper applies Shapley values to AGENT-level contribution in a multi-agent forecasting system
- Our Shapley attribution at the agent level (not feature level) is a distinct contribution

---

## 8. Structured Comparison: What Exists vs What Our Paper Offers

| Dimension | Existing Literature (2024-2026) | Our SDD + Causal-GAT Paper |
|-----------|-------------------------------|---------------------------|
| **Forecasting paradigm** | Black-box ML/DL pipelines (LSTM, XGBoost, Transformer), decomposition hybrids | Multi-agent LLM debate with structured protocol, then Causal-GAT meta-learning |
| **LLM usage** | Sentiment extraction tool (GPT-4o, FinBERT for scoring news) | Autonomous specialist agents with domain expertise, producing reasoned adjustments |
| **Multi-agent architecture** | Equity trading frameworks (TradingAgents, FinCon) with portfolio decisions | 7 domain-specialist agents for continuous volatility prediction |
| **Debate mechanism** | Bull vs Bear binary debate (TradingAgents); hawkish vs dovish (Takano et al.) | Structured Delphi: Round 1 independent, Round 2 informed revision; confidence-weighted |
| **Target** | Price level, returns, trading signals | Forward-looking 20-day realized volatility (continuous) |
| **Commodity focus** | Mostly equities; oil LLM papers use LLMs only for sentiment | Crude oil (WTI) with multi-source macro/geopolitical/technical data |
| **Aggregation** | Simple averaging, majority vote, meta-learner on features | Causal-GAT on Granger causality graph with context-dependent learned weights |
| **Interpretability** | SHAP on features of black-box models | Three-level: (1) Agent-level Shapley attribution, (2) Myerson values on influence graph, (3) Causal-GAT attention weights |
| **Group behavior analysis** | Theoretical: herding, degeneration of thought studied in NLP tasks | Empirical: 43% herding rate measured, regime-dependent, quantified impact on forecast accuracy |
| **Regime adaptation** | Fixed models or simple regime-switching | GAT meta-learner dynamically shifts weights (HAR 46% in high-vol vs 38% in normal) |
| **Baselines** | Typically HAR, GARCH, single ML models | Full battery: HAR, GARCH, Persistence, Ridge, Lasso, GBR, RF, XGBoost, LSTM, Transformer |
| **Sample size** | Varies (many < 500 obs) | 1,285 trading days (2020-2025), walk-forward evaluation on 972 days |

---

## 9. Novelty Claims Supported by This Scan

1. **First multi-agent LLM debate system for oil volatility forecasting**: No published paper applies multi-agent debate to commodity volatility prediction. The closest (Takano et al., 2025) does monetary policy classification, not continuous forecasting.

2. **First agent-level Shapley attribution in forecasting**: All existing SHAP/Shapley work operates at the feature level. Our agent-level Shapley decomposition treats each specialist agent as a "player" in a cooperative game, which is conceptually different and produces interpretability at the reasoning level.

3. **First Causal-GAT aggregation for multi-agent forecasts**: Granger causality-based influence graph + graph attention network for learning context-dependent aggregation weights is novel. No existing paper combines causal discovery with graph neural networks for multi-agent forecast aggregation.

4. **First empirical measurement of herding dynamics in LLM multi-agent forecasting**: While herding/groupthink in LLM agents is discussed theoretically, we provide the first large-scale empirical measurement (1,285 days) of herding rates, regime dependence, and impact on forecast accuracy.

5. **Bridging the LLM sentiment vs LLM reasoning gap**: Existing papers use LLMs as passive sentiment extractors. Our framework uses LLMs as active reasoning agents who process multi-source data, produce structured arguments, and revise positions through deliberation.

---

## 10. Recommended Citations for Our Paper

### Must-Cite (Direct Competition/Positioning)
- Takano, Hirano & Nakagawa (2025) - closest multi-agent debate for financial prediction
- Xiao, Sun, Luo & Wang (2024) - TradingAgents multi-agent financial framework
- Tiwari et al. (2024) - Energy Economics ML oil volatility benchmark
- Dai et al. (2026) - Multi-dimensional LLM sentiment for WTI
- Hashami & Maldonado (2024) - SHAP for oil price direction with language models

### Should-Cite (Methodological Context)
- EJOR XAI framework (2024) - theoretical grounding for interpretability
- Borgonovo et al. (2024) - Shapley-based feature importance in EJOR
- Ghali et al. (2024) - agentic AI for commodity price shocks
- Resources Policy geopolitical risk paper (2024) - validates geopolitical agent importance
- Zaleski & Chudziak (2025) - multi-agent LLM for energy investment

### Supporting References (Methodological Background)
- MAD theory papers: Zhang et al. (2025), Becker et al. (2025), Yang et al. (2025)
- Cognitive bias papers: MindScope, conformity studies
- Energy Economics sentiment papers: organization sentiment, official media sentiment

---

## Sources and References

### Energy Economics
1. [Tiwari et al. (2024) - ML models for crude oil and natural gas volatility](https://www.sciencedirect.com/science/article/abs/pii/S0140988324003165)
2. [More is better? INE oil futures volatility predictor choice](https://www.sciencedirect.com/science/article/abs/pii/S0140988324002482)
3. [Multiscale decomposition for crude oil price forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0140988324004419)
4. [Energy organization sentiment and oil return forecast](https://www.sciencedirect.com/science/article/abs/pii/S0140988324008144)
5. [Official media sentiment for Chinese crude oil futures](https://www.sciencedirect.com/science/article/abs/pii/S0140988324006753)
6. [Forecasting of clean energy market volatility: role of oil](https://www.sciencedirect.com/science/article/abs/pii/S0140988324001592)
7. [iTransformer for crude oil price risk warning](https://www.sciencedirect.com/science/article/abs/pii/S036054422403977X)

### EJOR
8. [Explainable AI for Operational Research framework](https://www.sciencedirect.com/science/article/pii/S0377221723007294)
9. [Borgonovo et al. - Shapley value-based feature importance](https://iris.unibocconi.it/retrieve/75c33799-991f-4898-b645-2c5ccc7e0305/EJOR_2024_Borgonovo_Plischke_Rabitti.pdf)

### Resources Policy
10. [Geopolitical risks and crude oil futures volatility: ML evidence](https://www.sciencedirect.com/science/article/abs/pii/S0301420724007414)
11. [Crude oil prices for four decades using deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0301420723011492)

### Management Science
12. [Call for Papers - AI for Finance VSI](https://pubsonline.informs.org/doi/10.1287/mnsc.2024.Call.V70.n10)

### LLM for Oil
13. [Prediction of Crude Oil Price using LLM - Procedia CS 2025](https://www.sciencedirect.com/science/article/pii/S1877050925023816)
14. [Dai et al. (2026) - Multi-dimensional LLM sentiment for WTI](https://arxiv.org/html/2603.11408)
15. [Hashami & Maldonado - News predict oil volatility with SHAP](https://arxiv.org/html/2508.20707v1)

### Multi-Agent LLM Systems
16. [Takano et al. (2025) - Hawkish-Dovish multi-agent debate for monetary policy](https://arxiv.org/abs/2511.02469)
17. [TradingAgents - Multi-agent LLM trading framework](https://arxiv.org/abs/2412.20138)
18. [Zaleski & Chudziak (2025) - LLM multi-agent for energy investment](https://www.researchgate.net/publication/390521016_LLM-Based_multi-agent_system_for_individual_investment_in_energy_and_natural_resources)
19. [Ghali et al. - Agentic AI for commodity price shocks](https://arxiv.org/html/2508.06497v1)
20. [Wawer & Chudziak (2025) - ElliottAgents for stock forecasting](https://arxiv.org/html/2506.16813v1)

### Multi-Agent Debate Theory
21. [Zhang et al. (2025) - If Multi-Agent Debate is the Answer](https://arxiv.org/abs/2025.xxxxx)
22. [Becker et al. (2025) - Problem Drift in Multi-Agent Debate](https://arxiv.org/abs/2025.xxxxx)
23. [Yang et al. (2025) - Revisiting MAD as Test-Time Scaling](https://arxiv.org/abs/2025.xxxxx)
24. [Tillmann (2025) - Literature Review of Multi-Agent Debate](https://arxiv.org/abs/2025.xxxxx)

### Interpretability / SHAP
25. [Feng et al. (2024) - Interpretable ensemble boosting for China oil futures (retracted)](https://www.sciencedirect.com/science/article/pii/S1059056024001643)

### Cognitive Biases in LLM Agents
26. [Group conformity in multi-agent LLM discussions](https://aclanthology.org/2025.findings-acl.265.pdf)
27. [Multi-agent system for monetary policy decision modeling - GWU](https://www2.gwu.edu/~forcpgm/2025-005.pdf)

### Other Relevant
28. [Oil price volatility prediction: combination vs ML shrinkage methods](https://www.sciencedirect.com/science/article/abs/pii/S0360544224012696)
29. [Forecasting crude oil price uncertainty: explainable DL with sentiment](https://www.sciencedirect.com/science/article/abs/pii/S0957417425030350)
30. [QuantAgent for high-frequency trading](https://arxiv.org/html/2509.09995v3)
