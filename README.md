# Influence-Aware-Multi-Agent-DSS
Influence-Aware Multi-Agent Decision Support System using UNSW-NB15
# Influence-Aware Multi-Agent Decision Support System (IA-MAS)

An Influence-Aware Multi-Agent Decision Support System for cybersecurity threat analysis using the **UNSW-NB15 dataset**.

## 1. Overview

This project proposes an Influence-Aware Multi-Agent Decision Support System (IA-MAS) designed to improve collaborative decision-making among specialized machine learning agents.

Instead of depending on a single classifier, the system uses multiple feature-specialist agents that independently analyze network traffic and contribute predictions to a shared decision-making process.

The framework incorporates agent confidence, reliability, influence relationships, consensus mechanisms and self-correction to investigate how collaboration between agents affects decision quality and robustness.

## 2. Research Objectives

- Develop specialized agents for different cybersecurity feature groups.
- Investigate influence-aware collaboration between agents.
- Incorporate confidence and reliability into multi-agent decision-making.
- Evaluate different consensus mechanisms.
- Analyze system robustness under agent degradation.
- Investigate self-correction mechanisms for improving decision reliability.

## 3. System Architecture

The framework consists of the following major components:

### A. Specialized Agents

Four machine learning agents analyze different aspects of network traffic:

| Agent | Responsibility |
|---|---|
| Traffic Agent | Analyzes network traffic characteristics |
| Threat Agent | Focuses on threat-related characteristics |
| Behaviour Agent | Analyzes behavioural patterns |
| Risk Agent | Contributes risk-oriented predictions |

### B. Orchestrator

Coordinates the agents, collects their predictions and manages the decision-making workflow.

### C. Evidence Agent

Supports evidence-based decision-making using available evidence.

### D. Critic Agent

Evaluates agent contributions and supports critical assessment of decisions.

### E. Influence Graph

Represents relationships between agents and supports influence-aware decision-making.

### F. Consensus Engine

Combines agent predictions through different consensus strategies.

### G. Self-Correction

Implements a mechanism for evaluating and correcting decisions under configured safeguards.

## 4. Dataset

The project uses the **UNSW-NB15 cybersecurity dataset**.

The dataset contains network traffic records used for network intrusion detection research.

Included files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

The project also contains a sample evidence file under `datasets/evidence/`.

Dataset source: [UNSW-NB15 Dataset – UNSW Canberra Cyber](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Please follow the dataset provider's terms and citation requirements when redistributing or reusing the data.

## 5. Technology Stack

- Python
- Machine Learning
- Scikit-learn / Joblib
- Pandas and NumPy
- Multi-Agent Systems
- Graph-based Influence Modelling
- CSV-based Experimental Analysis

## 7. Experimental Evaluation

The project includes experiments and analysis scripts covering:

- Agent agreement and disagreement analysis
- Agent confidence calibration
- Reliability estimation
- Consensus mechanism comparison
- Confidence-aware consensus
- Influence ablation studies
- Agent degradation experiments
- Dynamic adaptation and robustness evaluation
- Self-correction evaluation


## 8. Future Scope

- Improved adaptive influence estimation.
- More robust consensus mechanisms.
- Advanced agent reliability modelling.
- Improved self-correction strategies.
- Evaluation across additional cybersecurity datasets.
- Deployment in real-time intrusion detection environments.

## 9. Author and Repository

**GitHub:** [@anamikaa17](https://github.com/anamikaa17)

**Repository:** [Influence-Aware-Multi-Agent-DSS](https://github.com/anamikaa17/Influence-Aware-Multi-Agent-DSS)

---

*This repository contains the implementation and experimental artifacts for an ongoing research project on Influence-Aware Multi-Agent Decision Support Systems.*
