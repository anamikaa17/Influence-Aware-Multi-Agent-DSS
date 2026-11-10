
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph
from influence.confidence_aware_influence import ConfidenceAwareInfluence


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

print("\nLOADING TEST DATA")
df = pd.read_csv(TEST_PATH)

y_true = df["label"].astype(int).to_numpy()

runner = RealAgentRunner()
graph = DynamicInfluenceGraph()
confidence_engine = ConfidenceAwareInfluence(graph)

agent_names = list(AGENT_FEATURES.keys())

# Load validation-based historical reliability
with open("config/agent_reliability.json", "r") as file:
    reliability_data = json.load(file)

reliability = {
    agent: reliability_data[agent]["accuracy"]
    for agent in agent_names
}

# Generate specialist predictions and probabilities
predictions = {}
probabilities = {}

print("\nGENERATING SPECIALIST PREDICTIONS")

for agent, features in AGENT_FEATURES.items():

    model = runner.models[agent]
    X = df[features]

    predictions[agent] = model.predict(X)
    probabilities[agent] = model.predict_proba(X)

    print(f"{agent}: Completed")


# Get graph influence factors
outgoing_influence = {}

for agent in agent_names:

    outgoing_influence[agent] = sum(
        data["influence"]
        for _, _, data in graph.graph.out_edges(
            agent,
            data=True
        )
    )

max_influence = max(outgoing_influence.values())

graph_factors = {
    agent: (
        outgoing_influence[agent] / max_influence
        if max_influence > 0 else 1.0
    )
    for agent in agent_names
}

# Calculate confidence reliability for each sample
confidence_factors = {}

for agent in agent_names:

    confidence_factors[agent] = np.array([
        confidence_engine.get_confidence_adjustment(
            agent,
            float(np.max(probability))
        )
        for probability in probabilities[agent]
    ])


# Build four ablation configurations
configurations = {
    "Reliability Only": {
        "reliability": True,
        "graph": False,
        "confidence": False
    },
    "Graph Influence Only": {
        "reliability": False,
        "graph": True,
        "confidence": False
    },
    "Confidence Reliability Only": {
        "reliability": False,
        "graph": False,
        "confidence": True
    },
    "Full Proposed IA-MAS": {
        "reliability": True,
        "graph": True,
        "confidence": True
    }
}

results = []

print("\nINFLUENCE ABLATION STUDY")
print("=" * 80)

for configuration_name, settings in configurations.items():

    print(f"\nEvaluating: {configuration_name}")

    raw_weights = {}

    for agent in agent_names:

        # Begin with neutral factors
        reliability_factor = (
            reliability[agent]
            if settings["reliability"] else 1.0
        )

        graph_factor = (
            0.5 + 0.5 * graph_factors[agent]
            if settings["graph"] else 1.0
        )

        if settings["confidence"]:
            confidence_factor = confidence_factors[agent]
        else:
            confidence_factor = np.ones(len(df))

        raw_weights[agent] = (
            reliability_factor
            * graph_factor
            * confidence_factor
        )

    # Normalize weights separately for every sample
    total_weights = np.zeros(len(df))

    for agent in agent_names:
        total_weights += raw_weights[agent]

    total_weights = np.maximum(total_weights, 1e-12)

    normal_scores = np.zeros(len(df))
    attack_scores = np.zeros(len(df))

    for agent in agent_names:

        model_classes = list(runner.models[agent].classes_)

        normal_index = model_classes.index(0)
        attack_index = model_classes.index(1)

        normalized_weight = (
            raw_weights[agent] / total_weights
        )

        normal_scores += (
            normalized_weight
            * probabilities[agent][:, normal_index]
        )

        attack_scores += (
            normalized_weight
            * probabilities[agent][:, attack_index]
        )

    y_pred = (attack_scores > normal_scores).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    results.append({
        "Configuration": configuration_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "FNR": fnr
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")


# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/influence_ablation_results.csv",
    index=False
)

print("\nABLATION SUMMARY")
print("=" * 80)
print(results_df.to_string(index=False))

print("\nResults saved to:")
print("results/influence_ablation_results.csv")

print("\nABLATION STUDY COMPLETED")
