# Agent-Based Models for Financial Crisis and Extreme Events

Literature survey conducted 2026-03-18.

---

## 1. LLM-Powered Financial Market Simulation

### TwinMarket (Yang et al., 2025) - Behavioral and Social Simulation
- **arXiv:** 2502.01506
- **Venue:** ICLR 2025 Workshop
- **Method:** LLM agents with BDI model (rational decision-making, technical analysis, behavioral biases including herding and overconfidence). Scales to 1000 agents with social network.
- **Findings:** Reproduces fat-tailed distributions, volatility clustering, boom-bust cycles, self-fulfilling prophecy effects, information-driven crashes.
- **Key takeaway:** One of three test environments in Tang et al. (2026). Shows individual actions trigger emergent group behaviors.

### Social Contagion and Bank Runs (Ruano & Rajan, 2026)
- **arXiv:** 2602.15066
- **Method:** Process-based ABM with constrained LLM depositors. Heterogeneous risk tolerance. Communication on heavy-tailed network calibrated to Twitter/X from March 2023.
- **Findings:** Network connectivity amplifies withdrawal cascades. Cross-bank contagion shows phase transition near spillover rate 0.10. Reproduces SVB and First Republic ordering.
- **Key takeaway:** Network topology critically shapes crisis dynamics.

### Can LLMs Trade? (Lopez-Lira, 2025) - Testing Financial Theories
- **arXiv:** 2504.10789
- **Method:** Simulated market with persistent order book, market/limit orders, partial fills, dividends. LLMs as heterogeneous trading agents (value, momentum, market maker).
- **Findings:** LLMs follow assigned strategies. Markets exhibit realistic price discovery, bubbles, underreaction.

### FCLAgent (Hashimoto et al., 2025) - Fundamental-Chartist-LLM Agent
- **arXiv:** 2510.12189
- **Method:** LLMs for buy/sell decisions; rule-based methods for pricing and volume. Integrates LLMs into conventional ABM market simulations.
- **Findings:** FCLAgents reproduce path-dependent patterns that conventional agents miss.

---

## 2. Crisis Detection and Risk Frameworks

### Black Swan Dynamics (Sujatha et al., 2025) - Network-Based Systemic Risk
- **Venue:** Risk Management (Springer)
- **DOI:** 10.1057/s41283-025-00177-5
- **Method:** Complex network theory + extreme value statistics + computational linguistics. Systemic Vulnerability Index combining network fragility, tail risk, sentiment indicators.
- **Findings:** Risk elevation signals 1-5 months before systemic events (2008, Flash Crash, COVID-19).
- **Key takeaway:** Graph/network approaches to systemic risk detection are well-established.

### Enhancing Anomaly Detection with LLM Multi-Agent (Park, 2024)
- **arXiv:** 2403.19735
- **Method:** Collaborative AI agent network for S&P 500 anomaly detection. Specialized agents for data conversion, expert analysis, cross-checking, report consolidation.
- **Findings:** Improved anomaly detection efficiency and accuracy, reduced human intervention.

---

## 3. Scalable Multi-Agent Financial Systems

### MASS (Guo et al., 2025) - Multi-Agent Simulation Scaling
- **arXiv:** 2505.10278
- **Method:** Backward optimization for learning optimal distribution of heterogeneous agents. Scales to 512 agents for portfolio construction.
- **Findings:** Scaling effect: more agents yield improved excess returns.

### MARS (2025) - Meta-Adaptive Risk-Aware Multi-Agent Portfolio
- **arXiv:** 2508.01173
- **Method:** Heterogeneous Agent Ensemble with explicit risk profiles via Safety-Critic networks. Meta-Adaptive Controller shifts between conservative/aggressive agents based on conditions.
- **Findings:** Outperforms traditional and DRL baselines in risk-adjusted returns during volatile periods.

---

## 4. Surveys

### ABM in Economics and Finance: Past, Present, Future
- **Venue:** Journal of Economic Literature (AEA)
- **Scope:** Comprehensive survey. Bank of Canada adoption, calibration to Italian/OECD economies. All sectors with heterogeneous agents.
- **Key insight:** ABMs are "coming of age" with practical use in central banking and policy.

### LLM Agent in Financial Trading: A Survey (Ding et al., 2024)
- **arXiv:** 2408.06361
- **Scope:** 27 papers. Categorizes architectures into news-driven, reflection-driven, debate-driven, and RL-driven.

### MIRAI: Evaluating LLM Agents for Event Forecasting (2024)
- **arXiv:** 2407.01231
- **Method:** Benchmark with 991,759 GDELT event records. 705 query-answer pairs for forecasting evaluation.
- **Key takeaway:** Uses GDELT data for LLM agent forecasting evaluation, directly relevant to our GDELT-based approach.
