
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from agents.ml_agents import MLAgentTrainer
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLOADING DATA")

df = pd.read_csv(
    "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
)

X = df.drop(columns=["label", "attack_cat", "id"])
y = df["label"].astype(int).to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    stratify=y,
    random_state=42
)

X_cal, X_eval, y_cal, y_eval = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Training:", len(y_train))
print("Calibration:", len(y_cal))
print("Evaluation:", len(y_eval))


# --------------------------------------------------
# 2. TRAIN SPECIALIST AGENTS
# --------------------------------------------------

print("\nTRAINING SPECIALIST AGENTS")

trainer = MLAgentTrainer()
models = {}

agent_names = list(AGENT_FEATURES.keys())

for agent, features in AGENT_FEATURES.items():

    model = trainer.prepare_model(features, df)

    model.fit(
        X_train[features],
        y_train
    )

    models[agent] = model

    print(f"{agent}: Training completed")


# --------------------------------------------------
# 3. CALIBRATION PREDICTIONS
# --------------------------------------------------

print("\nGENERATING CALIBRATION PREDICTIONS")

cal_predictions = {}

for agent, features in AGENT_FEATURES.items():

    cal_predictions[agent] = models[agent].predict(
        X_cal[features]
    )


# --------------------------------------------------
# 4. LEARN INFLUENCE GRAPH
# --------------------------------------------------

print("\nLEARNING INFLUENCE GRAPH")

graph = DynamicInfluenceGraph()

graph.estimate_influence_from_predictions(
    y_cal,
    cal_predictions
)

graph.display_graph()


# Calculate normalized outgoing influence
outgoing = {}

for agent in agent_names:

    outgoing[agent] = sum(
        data["influence"]
        for _, _, data in graph.graph.out_edges(
            agent,
            data=True
        )
    )

max_outgoing = max(outgoing.values(), default=0)

graph_factors = {
    agent: (
        outgoing[agent] / max_outgoing
        if max_outgoing > 0 else 1.0
    )
    for agent in agent_names
}


# --------------------------------------------------
# 5. EVALUATION PREDICTIONS
# --------------------------------------------------

print("\nGENERATING EVALUATION PREDICTIONS")

eval_predictions = {}

for agent, features in AGENT_FEATURES.items():

    eval_predictions[agent] = models[agent].predict(
        X_eval[features]
    )

    print(f"{agent}: Completed")


# --------------------------------------------------
# 6. CONSENSUS METHODS
# --------------------------------------------------

def majority_vote(predictions):

    matrix = np.column_stack(
        list(predictions.values())
    )

    return (
        matrix.sum(axis=1) > matrix.shape[1] / 2
    ).astype(int)


def influence_vote(predictions):

    weights = {}

    for agent in agent_names:

        reliability = graph.graph.nodes[agent].get(
            "reliability", 1.0
        )

        weights[agent] = (
            reliability *
            (0.5 + 0.5 * graph_factors[agent])
        )

    total = sum(weights.values())

    weights = {
        agent: value / total
        for agent, value in weights.items()
    }

    scores = np.zeros(len(y_eval))

    for agent in agent_names:

        scores += (
            weights[agent] *
            predictions[agent]
        )

    return (scores > 0.5).astype(int)


# --------------------------------------------------
# 7. RELIABILITY DEGRADATION EXPERIMENT
# --------------------------------------------------

print("\nAGENT RELIABILITY DEGRADATION EXPERIMENT")

levels = [0, 0.10, 0.20, 0.30, 0.40, 0.50]

results = []

for target_agent in agent_names:

    print(f"\nDEGRADING: {target_agent}")

    for level in levels:

        modified = {
            agent: pred.copy()
            for agent, pred in eval_predictions.items()
        }

        if level > 0:

            rng = np.random.default_rng(42)

            count = int(len(y_eval) * level)

            indices = rng.choice(
                len(y_eval),
                size=count,
                replace=False
            )

            modified[target_agent][indices] = (
                1 - modified[target_agent][indices]
            )

        majority = majority_vote(modified)

        influence = influence_vote(modified)

        for method, prediction in [
            ("Majority Voting", majority),
            ("Influence-Aware", influence)
        ]:

            accuracy = accuracy_score(
                y_eval, prediction
            )

            macro_f1 = f1_score(
                y_eval,
                prediction,
                average="macro",
                zero_division=0
            )

            results.append({
                "degraded_agent": target_agent,
                "degradation": level,
                "method": method,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

            print(
                f"{level:.0%} | {method}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}"
            )


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/agent_degradation_results.csv",
    index=False
)

print("\nRESULTS SAVED:")
print("results/agent_degradation_results.csv")

print("\nEXPERIMENT COMPLETED")
