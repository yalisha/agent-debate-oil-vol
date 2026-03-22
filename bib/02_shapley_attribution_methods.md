# Shapley Value Attribution in Multi-Agent Systems

Literature survey conducted 2026-03-18.

---

## 1. Core Paper: MAS Extreme Event Attribution

### Tang et al. (2026) - Interpreting Emergent Extreme Events in Multi-Agent Systems
- **arXiv:** 2601.20538
- **Affiliation:** Shanghai AI Laboratory
- **Method:** Shapley value decomposition of extreme event risk to individual agent actions. Monte Carlo approximation (M=1000). Counterfactual simulation replacing actions with safe baselines. Three aggregation dimensions: temporal, agent, behavioral.
- **Environments:** EconAgent (macroeconomic, 10 agents), TwinMarket (financial, 10 agents), SocialNetwork (polarization, 20 agents).
- **Five findings:** Risk latency, Pareto principle (G_ag > 0.4), instability correlation, herding effect (Z_ag > 0.3), toxic behaviors (G_be > 0.5).
- **Intervention:** Replace top-k attributed actions with safe baselines, re-simulate. Superior risk reduction vs Random/FT/FA/AT baselines.
- **GitHub:** github.com/mjl0613ddm/IEEE

---

## 2. Shapley Value in Multi-Agent RL (Credit Assignment)

### Shapley Q-value (Wang et al., 2020)
- **Venue:** AAAI 2020
- **arXiv:** 1907.05707
- **Method:** Extended convex game framework. SQDDPG algorithm using Shapley Q-value as critic for fair reward distribution.
- **Findings:** Improved convergence over MADDPG and COMA.

### Shapley Counterfactual Credits (Li et al., 2021)
- **Venue:** KDD 2021
- **arXiv:** 2106.00285
- **Method:** Shapley value for crediting agent combinations in deep MARL. Monte Carlo sampling for approximation.
- **Findings:** SOTA on StarCraft II benchmarks, especially high-difficulty tasks.

### SHAQ (Wang et al., 2022)
- **Venue:** NeurIPS 2022
- **arXiv:** 2105.15013
- **Method:** Markov Shapley Value (MSV) generalizing Shapley to Markov convex games. Shapley-Bellman optimality equation.
- **Findings:** Superior performance with theoretical grounding for value factorization.
- **Key takeaway:** Provides rigorous theoretical foundation for Shapley in sequential settings.

### Shapley Value Based MARL (Wang, 2024)
- **arXiv:** 2402.15324
- **Method:** Doctoral thesis extending convex games to MDP settings. Three algorithms: SHAQ, SQDDPG, SMFPPO. Extension to partial observability.
- **Findings:** Markov Shapley value satisfies efficiency, dummy identification, contribution reflection, symmetry.

### SHARP (Li et al., 2026)
- **arXiv:** 2602.08335
- **Method:** Shapley-based credit + hierarchical planner-worker optimization. Decomposes rewards into global broadcast, Shapley marginal-credit, and tool-process components.
- **Findings:** 23.66% and 14.05% average improvement over single-agent and multi-agent baselines.

---

## 3. Myerson Value and Graph-Restricted Shapley

### L-Shapley and C-Shapley (Chen et al., 2018)
- **Venue:** ICLR 2019
- **arXiv:** 1808.02610
- **Method:** Linear-complexity algorithms for feature importance when contributions follow graph-structured factorization. C-Shapley of full order = Myerson value.
- **Findings:** Comparable to full Shapley with linear cost.
- **Key takeaway:** Establishes the formal Myerson-Shapley connection for structured data.

### Monte Carlo for Myerson Value (Tarkowski et al., 2020)
- **arXiv:** 2001.00065
- **Method:** Three MC approaches: conventional permutation, hybrid exact+sampling, connected coalition sampling.
- **Findings:** Hybrid algorithm significantly improves approximation quality.
- **Key takeaway:** Provides practical computation methods for Myerson value on arbitrary graphs.

### Myerson Values for Cooperative MAS Explanation (Angelotti & Diaz-Rodriguez, 2023)
- **Venue:** Knowledge-Based Systems, Vol. 260
- **arXiv:** 2212.03041
- **Method:** Shapley and Myerson analysis for agent policy and attribute contributions. Hierarchical Knowledge Graph for dynamic programming optimization.
- **Findings:** Myerson values are efficient alternative to Shapley by exploiting graph structure.
- **Key takeaway:** MOST DIRECTLY RELEVANT. Demonstrates Myerson value for MAS explainability with graph structure.

### Myerson Values for GNN Interpretation (Harries et al., 2024)
- **Venue:** J. Chemical Information and Modeling
- **Method:** Myerson values for GNN predictions treating GNN as coalition game with molecular graph as cooperation structure.
- **Findings:** Node attributions constrained by graph connectivity.
- **GitHub:** github.com/kochgroup/myerson (Python package)

---

## 4. Graph-Based Shapley for GNN Explainability

### GraphSVX (Duval & Malliaros, 2021)
- **Venue:** ECML PKDD 2021
- **arXiv:** 2104.10482
- **Method:** Post-hoc model-agnostic GNN explanation using surrogate model on perturbed dataset with Shapley values.

### GNNShap (Akkas & Azad, 2024)
- **Venue:** WWW 2024
- **arXiv:** 2401.04829
- **Method:** Edge-level Shapley explanations for GNNs. GPU-parallelized sampling with pruning.
- **Findings:** Faster and better fidelity than GNNExplainer, PGMExplainer, GraphSVX.

---

## 5. Causal Shapley Values

### Causal Shapley Values (Heskes et al., 2020)
- **Venue:** NeurIPS 2020
- **arXiv:** 2011.01625
- **Method:** Causal Shapley values using Pearl's do-calculus. Separates direct and indirect effects.
- **Findings:** Better alignment with user intuition when features are dependent.

### cc-Shapley (Martin & Haufe, 2026)
- **arXiv:** 2602.20396
- **Method:** Interventional modification leveraging causal structure to eliminate collider-bias-induced spurious associations.
- **Findings:** Standard Shapley can produce misleading attributions; causal context fixes this.

---

## 6. Shapley for MAS Explainability

### Collective XAI (Heuillet et al., 2022)
- **Venue:** IEEE Computational Intelligence Magazine
- **arXiv:** 2110.01307
- **Method:** Monte Carlo Shapley values for agent contributions in cooperative MARL.
- **Findings:** Shapley values grow proportionally to capability improvements. Limitation: model-level not episode-level.

### Rollout-based Shapley Values (Ruggeri et al., 2024)
- **Venue:** IEEE ICMLCN 2024
- **Method:** MC approximation of Local Shapley Values via rollouts. Model-agnostic local and global explanations.
- **Findings:** Provides episode-level explanations that prior global methods could not.

### Counterfactual Shapley Values for RL (Shi et al., 2024)
- **arXiv:** 2408.02529
- **Method:** Counterfactual Difference Characteristic Value integrating counterfactual analysis with Shapley.
- **Findings:** Enhanced transparency, quantifies differences across decisions more effectively.

---

## 7. LLM Multi-Agent Attribution

### HiveMind (Xia et al., 2025) - DAG-Shapley
- **arXiv:** 2512.06432
- **Venue:** AAAI 2026
- **Method:** DAG-Shapley leverages DAG structure of agent workflows to prune non-viable coalitions. Contribution-Guided Online Prompt Optimization.
- **Findings:** Reduces LLM calls by 80%+ while maintaining attribution accuracy. Superior in multi-agent stock-trading.
- **Key takeaway:** HIGHLY RELEVANT. DAG-Shapley is a direct precedent for graph-structured Shapley in LLM agent systems.

### AgentSHAP (Horovicz, 2025) - Tool Importance in LLM Agents
- **arXiv:** 2512.12597
- **Method:** Model-agnostic Monte Carlo Shapley for tool attribution in LLM agents as black boxes.
- **Findings:** First explainability method for tool attribution in LLM agents.

---

## 8. Shapley in Ensemble Forecasting

### SVDE (2025) - Shapley Value Dynamic Ensemble for Load Forecasting
- **Venue:** Int. J. Electrical Power and Energy Systems
- **Method:** Cooperative game-theoretic ensemble where models are players. Shapley-driven dynamic weighting.
- **Findings:** 20%+ improvement over baselines and static weighting.
- **Key takeaway:** Direct precedent for Shapley-weighted ensemble prediction.

### Beyond Forecast Leaderboards (2025)
- **Venue:** ScienceDirect
- **Method:** Shapley as game-theoretic measure of model's marginal contribution to ensemble accuracy.
- **Findings:** Leaderboard rankings do not capture true model contribution.

---

## 9. Surveys

### The Shapley Value in Machine Learning (Rozemberczki et al., 2022)
- **Venue:** IJCAI 2022
- **arXiv:** 2202.05594
- **Scope:** Feature selection, explainability, MARL, ensemble pruning, data valuation.
- **GitHub:** github.com/AstraZeneca/awesome-shapley-value

### Beyond Shapley Values (Il Idrissi et al., 2025)
- **Venue:** IJCAI 2025 Workshop on XAI
- **arXiv:** 2506.13900
- **Key insight:** Weber and Harsanyi sets offer richer alternatives to Shapley for attribution.
