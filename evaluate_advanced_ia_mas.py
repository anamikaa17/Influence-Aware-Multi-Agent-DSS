
import json
import os
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
from consensus.consensus_engine import ConsensusEngine


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
RELIABILITY_PATH = "config/agent_reliability.json"
RESULT_PATH = "results/advanced_ia_mas_evaluation.csv"


# --------------------------------------------------
# 1. LOAD TEST DATA AND VALIDATION RELIABILITY
# --------------------------------------------------

print("\nLOADING TEST DATA")

df = pd.read_csv(TEST_PATH)
y_true = df["label"].astype(int).to_numpy()

with open(RELIABILITY_PATH, "r") as file:
    validation_reliability = json.load(file)

runner = RealAgentRunner()
graph = DynamicInfluenceGraph()

agent_names = list(AGENT_FEATURES.keys())

# Apply validation-derived reliability to graph
for agent in agent_names:

    if agent not in validation_reliability:
        raise ValueError(
            f"Missing validation reliability for {agent}"
        )

    values = validation_reliability[agent]

    node = graph.graph.nodes[agent]

    node["reliability"] = float(
        values["overall_reliability"]
    )

    node["normal_precision"] = float(
        values["normal_precision"]
    )

    node["attack_precision"] = float(
        values["attack_precision"]
    )

graph.update_influence_weights()

confidence_engine = ConfidenceAwareInfluence(graph)
consensus_engine = ConsensusEngine(graph)


# --------------------------------------------------
# 2. GENERATE SPECIALIST PREDICTIONS
# --------------------------------------------------

predictions = {}
probabilities = {}

print("\nGENERATING SPECIALIST PREDICTIONS")

for agent in agent_names:

    model = runner.models[agent]
    X = df[AGENT_FEATURES[agent]]

    predictions[agent] = model.predict(X)

    raw_probabilities = model.predict_proba(X)

    class_indices = list(model.classes_)

    probabilities[agent] = np.column_stack([
        raw_probabilities[:, class_indices.index(0)],
        raw_probabilities[:, class_indices.index(1)]
    ])

    print(f"{agent}: Completed")


# --------------------------------------------------
# 3. METHOD 1: MAJORITY VOTING
# --------------------------------------------------

prediction_matrix = np.column_stack([
    predictions[agent] for agent in agent_names
])

majority_predictions = (
    prediction_matrix.sum(axis=1) > len(agent_names) / 2
).astype(int)


# --------------------------------------------------
# 4. METHOD 2: STATIC WEIGHTED VOTING
# --------------------------------------------------

static_weights = consensus_engine.static_weights

static_scores = np.zeros(len(df))

for agent in agent_names:

    static_scores += (
        predictions[agent] * static_weights[agent]
    )

static_predictions = (
    static_scores > sum(static_weights.values()) / 2
).astype(int)


# --------------------------------------------------
# 5. METHOD 3: DYNAMIC HARD VOTING
# --------------------------------------------------

dynamic_weights = consensus_engine.calculate_dynamic_weights(
    {agent: 0 for agent in agent_names}
)

dynamic_scores = np.zeros(len(df))

for agent in agent_names:

    dynamic_scores += (
        predictions[agent] * dynamic_weights[agent]
    )

dynamic_predictions = (
    dynamic_scores > 0.5
).astype(int)


# --------------------------------------------------
# 6. METHOD 4: PREVIOUS CONFIDENCE-AWARE IA-MAS
# --------------------------------------------------

normal_scores = np.zeros(len(df))
attack_scores = np.zeros(len(df))

for agent in agent_names:

    normal_scores += (
        dynamic_weights[agent] * probabilities[agent][:, 0]
    )

    attack_scores += (
        dynamic_weights[agent] * probabilities[agent][:, 1]
    )

previous_confidence_predictions = (
    attack_scores > normal_scores
).astype(int)


# --------------------------------------------------
# 7. METHOD 5: ADVANCED INTERPOLATED IA-MAS
# --------------------------------------------------

print("\nCALCULATING ADVANCED INFLUENCE WEIGHTS")

advanced_normal_scores = np.zeros(len(df))
advanced_attack_scores = np.zeros(len(df))

for index in range(len(df)):

    sample_probabilities = {
        agent: probabilities[agent][index]
        for agent in agent_names
    }

    weights = confidence_engine.calculate_weights(
        sample_probabilities
    )

    for agent in agent_names:

        advanced_normal_scores[index] += (
            weights[agent] * sample_probabilities[agent][0]
        )

        advanced_attack_scores[index] += (
            weights[agent] * sample_probabilities[agent][1]
        )

    if (index + 1) % 10000 == 0:
        print(f"Processed {index + 1} records")

advanced_predictions = (
    advanced_attack_scores > advanced_normal_scores
).astype(int)


# --------------------------------------------------
# 8. METHOD 6: CONTEXT-AWARE IA-MAS
# --------------------------------------------------

print("\nCALCULATING CONTEXT-AWARE IA-MAS")

# Graph influence factors are fixed for this set of agents.
influence_factors = consensus_engine._get_influence_factors(
    agent_names
)

context_raw_weights = {}

for agent in agent_names:

    node = graph.graph.nodes[agent]

    context_raw_weights[agent] = (
        np.where(
            predictions[agent] == 0,
            node["normal_precision"],
            node["attack_precision"]
        )
        * (
            0.5 + 0.5 * influence_factors[agent]
        )
    )

context_weight_matrix = np.column_stack([
    context_raw_weights[agent]
    for agent in agent_names
])

context_weight_sums = context_weight_matrix.sum(axis=1)

context_weight_matrix = (
    context_weight_matrix
    / np.maximum(context_weight_sums[:, None], 1e-12)
)

context_scores = np.zeros((len(df), 2))

for index, agent in enumerate(agent_names):

    context_scores[:, 0] += (
        context_weight_matrix[:, index]
        * (predictions[agent] == 0)
    )

    context_scores[:, 1] += (
        context_weight_matrix[:, index]
        * (predictions[agent] == 1)
    )

context_predictions = (
    context_scores[:, 1] > context_scores[:, 0]
).astype(int)


# --------------------------------------------------
# 9. METHOD 7: CONTEXT-AWARE CONFIDENCE IA-MAS
# --------------------------------------------------

print("\nCALCULATING CONTEXT-AWARE CONFIDENCE IA-MAS")

context_confidence_scores = np.zeros((len(df), 2))

for index, agent in enumerate(agent_names):

    context_confidence_scores[:, 0] += (
        context_weight_matrix[:, index]
        * probabilities[agent][:, 0]
    )

    context_confidence_scores[:, 1] += (
        context_weight_matrix[:, index]
        * probabilities[agent][:, 1]
    )

context_confidence_predictions = (
    context_confidence_scores[:, 1]
    > context_confidence_scores[:, 0]
).astype(int)


# --------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------

methods = {
    "Majority Voting MAS": majority_predictions,
    "Static Weighted MAS": static_predictions,
    "Dynamic Hard Voting MAS": dynamic_predictions,
    "Previous Confidence-Aware IA-MAS": previous_confidence_predictions,
    "Advanced Interpolated IA-MAS": advanced_predictions,
    "Context-Aware IA-MAS": context_predictions,
    "Context-Aware Confidence IA-MAS": context_confidence_predictions
}

results = []

print("\nFULL TEST DATASET EVALUATION")
print("=" * 85)

for method_name, y_pred in methods.items():

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    results.append({
        "Method": method_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "FNR": fnr
    })

    print(f"\n{method_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")


# --------------------------------------------------
# 11. DECISION COMPARISON
# --------------------------------------------------

print("\nDECISION COMPARISON")
print("=" * 60)

for method_name, y_pred in methods.items():

    different = int(
        np.sum(y_pred != majority_predictions)
    )

    print(f"\n{method_name} vs Majority Voting")
    print("Different decisions:", different)
    print("Identical decisions:", len(df) - different)


# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nRESULTS SAVED TO:")
print(RESULT_PATH)

print("\nCONTEXT-AWARE IA-MAS EVALUATION COMPLETED")
