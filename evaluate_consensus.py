
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from agents.real_agent_runner import RealAgentRunner
from consensus.consensus_engine import ConsensusEngine
from influence.influence_graph import DynamicInfluenceGraph


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
RELIABILITY_PATH = "config/agent_reliability.json"


def calculate_metrics(y_true, predictions):

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Macro F1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "FPR": fp / (fp + tn) if (fp + tn) else 0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0
    }


print("\nLOADING TEST DATA")

df = pd.read_csv(TEST_PATH)
y_true = df["label"].astype(int).to_numpy()

# Load the trained ML models
runner = RealAgentRunner()

# Create influence graph
graph = DynamicInfluenceGraph()

# Load validation-based reliability
with open(RELIABILITY_PATH, "r") as file:
    reliability_data = json.load(file)

print("\nLOADING VALIDATION RELIABILITY")

for agent_name, values in reliability_data.items():

    accuracy = values["accuracy"]

    graph.graph.nodes[agent_name]["reliability"] = accuracy

    print(f"{agent_name}: {accuracy:.4f}")

# Recalculate influence weights after reliability updates
graph.update_influence_weights()

# Initialise consensus engine
engine = ConsensusEngine(graph)

# Generate individual agent predictions
agent_predictions = {}

print("\nGENERATING TEST PREDICTIONS")

for agent_name in runner.models:

    from agents.feature_groups import AGENT_FEATURES

    features = AGENT_FEATURES[agent_name]

    agent_predictions[agent_name] = runner.models[
        agent_name
    ].predict(df[features])

    print(f"{agent_name}: Completed")

# Generate consensus predictions
majority_predictions = []
static_predictions = []
proposed_predictions = []

print("\nRUNNING CONSENSUS METHODS")

for i in range(len(df)):

    opinions = {
        agent: int(predictions[i])
        for agent, predictions in agent_predictions.items()
    }

    majority_predictions.append(
        engine.majority_voting(opinions)
    )

    static_predictions.append(
        engine.static_weighted(opinions)
    )

    proposed_predictions.append(
        engine.proposed_consensus(opinions)
    )

# Calculate performance
results = {}

for agent_name, predictions in agent_predictions.items():

    results[agent_name] = calculate_metrics(
        y_true, predictions
    )

results["Majority Voting MAS"] = calculate_metrics(
    y_true, majority_predictions
)

results["Static Weighted MAS"] = calculate_metrics(
    y_true, static_predictions
)

results["Proposed IA-MAS"] = calculate_metrics(
    y_true, proposed_predictions
)

results_df = pd.DataFrame(results).T

print("\n" + "=" * 75)
print("VALIDATION-CALIBRATED CONSENSUS EVALUATION")
print("=" * 75)

print(results_df.round(4).to_string())

# Compare static and proposed decisions
static_array = pd.Series(static_predictions).to_numpy()
proposed_array = pd.Series(proposed_predictions).to_numpy()

print("\nDECISION DIFFERENCE ANALYSIS")

print(
    "Different decisions:",
    (static_array != proposed_array).sum()
)

print(
    "Identical decisions:",
    (static_array == proposed_array).sum()
)

results_df.to_csv(
    "results/calibrated_consensus_evaluation.csv"
)

print("\nResults saved successfully.")
