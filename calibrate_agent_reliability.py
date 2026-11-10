
import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from agents.ml_agents import MLAgentTrainer, TRAIN_PATH
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph
from consensus.consensus_engine import ConsensusEngine


print("\nCALIBRATING AGENT RELIABILITY")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD OFFICIAL TRAINING DATA ONLY
# --------------------------------------------------

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Independent stratified validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING RECORDS:", len(X_train))
print("VALIDATION RECORDS:", len(X_val))

trainer = MLAgentTrainer()
graph = DynamicInfluenceGraph()

validation_results = {}

# --------------------------------------------------
# 2. TRAIN AND VALIDATE ALL SPECIALIST AGENTS
# --------------------------------------------------

print("\nTRAINING AND VALIDATING SPECIALIST AGENTS")

for agent_name, features in AGENT_FEATURES.items():

    print("\n" + "-" * 55)
    print("AGENT:", agent_name)

    model = trainer.prepare_model(features, X_train)

    model.fit(
        X_train[features],
        y_train
    )

    predictions = model.predict(
        X_val[features]
    )

    # Overall validation metrics
    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------
    # 3. PREDICTION-SPECIFIC PRECISION
    # --------------------------------------------------

    normal_mask = predictions == 0
    attack_mask = predictions == 1

    normal_predictions = int(normal_mask.sum())
    attack_predictions = int(attack_mask.sum())

    normal_correct = int(
        ((predictions == 0) & (y_val.to_numpy() == 0)).sum()
    )

    attack_correct = int(
        ((predictions == 1) & (y_val.to_numpy() == 1)).sum()
    )

    # Laplace-smoothed prediction precision
    normal_precision = (
        (normal_correct + 1) / (normal_predictions + 2)
        if normal_predictions > 0 else None
    )

    attack_precision = (
        (attack_correct + 1) / (attack_predictions + 2)
        if attack_predictions > 0 else None
    )

    # --------------------------------------------------
    # 4. UPDATE INFLUENCE GRAPH
    # --------------------------------------------------

    graph_metrics = graph.update_prediction_reliability(
        agent_name=agent_name,
        y_true=y_val.to_numpy(),
        predictions=predictions
    )

    # --------------------------------------------------
    # 5. STORE VALIDATION RESULTS
    # --------------------------------------------------

    validation_results[agent_name] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),

        "normal_precision": (
            float(normal_precision)
            if normal_precision is not None else None
        ),

        "attack_precision": (
            float(attack_precision)
            if attack_precision is not None else None
        ),

        "normal_prediction_count": normal_predictions,
        "attack_prediction_count": attack_predictions,

        "normal_correct_predictions": normal_correct,
        "attack_correct_predictions": attack_correct,

        "overall_reliability": float(
            graph_metrics["overall_reliability"]
        )
    }

    print("\nVALIDATION RESULTS")

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print(
        "Normal Prediction Precision:",
        round(normal_precision, 4)
        if normal_precision is not None else "Unavailable"
    )

    print(
        "Attack Prediction Precision:",
        round(attack_precision, 4)
        if attack_precision is not None else "Unavailable"
    )

    print("Normal Predictions:", normal_predictions)
    print("Attack Predictions:", attack_predictions)

    print(
        "Overall Reliability:",
        round(graph_metrics["overall_reliability"], 4)
    )

# --------------------------------------------------
# 6. SAVE VALIDATION RELIABILITY
# --------------------------------------------------

os.makedirs("config", exist_ok=True)

with open(
    "config/agent_reliability.json",
    "w"
) as file:

    json.dump(
        validation_results,
        file,
        indent=4
    )

print("\nVALIDATION RELIABILITY SAVED")
print("config/agent_reliability.json")

# --------------------------------------------------
# 7. DISPLAY CONTEXTUAL RELIABILITY
# --------------------------------------------------

print("\n" + "=" * 65)
print("CONTEXT-AWARE AGENT RELIABILITY")
print("=" * 65)

graph.display_graph()

# --------------------------------------------------
# 8. CALCULATE DYNAMIC CONSENSUS WEIGHTS
# --------------------------------------------------

engine = ConsensusEngine(graph)

sample_opinions = {
    "Traffic Agent": 0,
    "Threat Agent": 1,
    "Behaviour Agent": 0,
    "Risk Agent": 1
}

dynamic_weights = engine.calculate_dynamic_weights(
    sample_opinions
)

contextual_weights = engine.calculate_contextual_weights(
    sample_opinions
)

print("\nORIGINAL DYNAMIC WEIGHTS")
print("=" * 50)

for agent, weight in dynamic_weights.items():
    print(f"{agent}: {weight:.4f}")

print(
    "Total:",
    round(sum(dynamic_weights.values()), 4)
)

print("\nCONTEXT-AWARE DYNAMIC WEIGHTS")
print("=" * 50)

for agent, weight in contextual_weights.items():
    print(f"{agent}: {weight:.4f}")

print(
    "Total:",
    round(sum(contextual_weights.values()), 4)
)

print("\nRELIABILITY CALIBRATION COMPLETED")
