
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from agents.ml_agents import MLAgentTrainer
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph


# --------------------------------------------------
# 1. LOAD AND SPLIT DATA
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

    model.fit(X_train[features], y_train)

    models[agent] = model

    print(f"{agent}: Training completed")


# --------------------------------------------------
# 3. CALIBRATION PREDICTIONS
# --------------------------------------------------

print("\nCALIBRATING AGENTS")

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

summary = graph.estimate_influence_from_predictions(
    y_cal,
    cal_predictions
)

base_reliability = {
    agent: summary[agent]["reliability"]
    for agent in agent_names
}

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
# 6. CONSENSUS FUNCTIONS
# --------------------------------------------------

def majority_vote(predictions):

    matrix = np.column_stack(
        [predictions[a] for a in agent_names]
    )

    return (
        matrix.sum(axis=1) > len(agent_names) / 2
    ).astype(int)


def weighted_vote(predictions, weights):

    total_weight = sum(weights.values())

    if total_weight <= 0:
        return majority_vote(predictions)

    scores = np.zeros(
        len(next(iter(predictions.values())))
    )

    for agent in agent_names:

        scores += (
            weights[agent] *
            predictions[agent]
        )

    return (
        scores > total_weight / 2
    ).astype(int)


# --------------------------------------------------
# 7. DYNAMIC ADAPTATION EXPERIMENT
# --------------------------------------------------

print("\nDYNAMIC AGENT RELIABILITY ADAPTATION")

BATCH_SIZE = 500
ALPHA = 0.30
DEGRADATION = 0.30

results = []

for target_agent in agent_names:

    print("\n" + "=" * 65)
    print("TARGET AGENT:", target_agent)
    print("=" * 65)

    # Start with calibration reliability
    adaptive_reliability = base_reliability.copy()

    fixed_weights = {
        agent: base_reliability[agent] *
        (0.5 + 0.5 * graph_factors[agent])
        for agent in agent_names
    }

    adaptive_predictions = []
    fixed_predictions = []
    majority_predictions = []

    # First half: degraded agent
    # Second half: agent returns to normal

    midpoint = len(y_eval) // 2

    rng = np.random.default_rng(42)

    corrupted_indices = set(
        rng.choice(
            midpoint,
            size=int(midpoint * DEGRADATION),
            replace=False
        ).tolist()
    )

    for start in range(0, len(y_eval), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(y_eval))

        batch_predictions = {
            agent: eval_predictions[agent][start:end].copy()
            for agent in agent_names
        }

        batch_labels = y_eval[start:end]

        # Simulate 30% prediction corruption
        # during the first half only

        if start < midpoint:

            for local_index, global_index in enumerate(
                range(start, end)
            ):

                if global_index in corrupted_indices:

                    batch_predictions[target_agent][
                        local_index
                    ] = 1 - batch_predictions[target_agent][
                        local_index
                    ]

        # Decisions are made BEFORE feedback updates

        fixed_batch = weighted_vote(
            batch_predictions,
            fixed_weights
        )

        adaptive_weights = {
            agent: adaptive_reliability[agent] *
            (0.5 + 0.5 * graph_factors[agent])
            for agent in agent_names
        }

        adaptive_batch = weighted_vote(
            batch_predictions,
            adaptive_weights
        )

        majority_batch = majority_vote(
            batch_predictions
        )

        fixed_predictions.extend(fixed_batch.tolist())
        adaptive_predictions.extend(adaptive_batch.tolist())
        majority_predictions.extend(majority_batch.tolist())

        # Delayed feedback: reveal labels only AFTER
        # the current batch decisions are completed.

        for agent in agent_names:

            batch_accuracy = accuracy_score(
                batch_labels,
                batch_predictions[agent]
            )

            adaptive_reliability[agent] = (
                (1 - ALPHA) * adaptive_reliability[agent]
                + ALPHA * batch_accuracy
            )

        if start < midpoint <= end:
            print("Reliability update at phase transition:")
            for agent in agent_names:
                print(
                    f"{agent}: "
                    f"{adaptive_reliability[agent]:.4f}"
                )

    # --------------------------------------------------
    # 8. EVALUATE
    # --------------------------------------------------

    predictions = {
        "Majority Voting": np.array(majority_predictions),
        "Fixed Influence": np.array(fixed_predictions),
        "Adaptive Influence": np.array(adaptive_predictions)
    }

    for method, pred in predictions.items():

        accuracy = accuracy_score(y_eval, pred)

        macro_f1 = f1_score(
            y_eval,
            pred,
            average="macro",
            zero_division=0
        )

        results.append({
            "target_agent": target_agent,
            "method": method,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        print(
            f"{method}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}"
        )


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/dynamic_adaptation_results.csv",
    index=False
)

print("\nRESULTS SAVED:")
print("results/dynamic_adaptation_results.csv")

print("\nDYNAMIC ADAPTATION EXPERIMENT COMPLETED")
