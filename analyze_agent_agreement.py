
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA AND GENERATE PREDICTIONS
# --------------------------------------------------

df = pd.read_csv(TEST_PATH)
y_true = df["label"].astype(int).to_numpy()

runner = RealAgentRunner()
agent_names = list(AGENT_FEATURES.keys())

predictions = {}

print("\nGENERATING AGENT PREDICTIONS")
print("=" * 65)

for agent in agent_names:

    model = runner.models[agent]

    predictions[agent] = model.predict(
        df[AGENT_FEATURES[agent]]
    )

    accuracy = accuracy_score(
        y_true,
        predictions[agent]
    )

    print(f"{agent}: Accuracy = {accuracy:.4f}")

prediction_matrix = np.column_stack([
    predictions[agent] for agent in agent_names
])

# Number of agents voting for each class
attack_votes = prediction_matrix.sum(axis=1)
normal_votes = len(agent_names) - attack_votes

agreement_count = np.maximum(
    attack_votes,
    normal_votes
)

# Majority prediction; ties are resolved as Normal (0)
majority_prediction = (
    attack_votes > normal_votes
).astype(int)


# --------------------------------------------------
# 2. PAIRWISE PREDICTION AGREEMENT
# --------------------------------------------------

print("\nPAIRWISE AGENT AGREEMENT")
print("=" * 65)

agreement_matrix = pd.DataFrame(
    index=agent_names,
    columns=agent_names,
    dtype=float
)

kappa_matrix = pd.DataFrame(
    index=agent_names,
    columns=agent_names,
    dtype=float
)

for agent_a in agent_names:

    for agent_b in agent_names:

        pred_a = predictions[agent_a]
        pred_b = predictions[agent_b]

        agreement_matrix.loc[agent_a, agent_b] = np.mean(
            pred_a == pred_b
        )

        kappa_matrix.loc[agent_a, agent_b] = cohen_kappa_score(
            pred_a,
            pred_b
        )

print("\nPrediction Agreement Matrix")
print(agreement_matrix.round(4))

print("\nCohen's Kappa Matrix")
print(kappa_matrix.round(4))


# --------------------------------------------------
# 3. AGENT ERROR CORRELATION
# --------------------------------------------------

print("\nAGENT ERROR CORRELATION")
print("=" * 65)

error_data = {}

for agent in agent_names:

    error_data[agent] = (
        predictions[agent] != y_true
    ).astype(int)

error_df = pd.DataFrame(error_data)

error_correlation = error_df.corr()

print(error_correlation.round(4))


# --------------------------------------------------
# 4. AGREEMENT LEVEL ANALYSIS
# --------------------------------------------------

print("\nAGREEMENT LEVEL VS ACTUAL ACCURACY")
print("=" * 65)

diagnostics = []

for count in [4, 3, 2]:

    mask = agreement_count == count

    sample_count = int(mask.sum())

    if sample_count == 0:
        continue

    group_true = y_true[mask]
    group_pred = majority_prediction[mask]

    accuracy = accuracy_score(
        group_true,
        group_pred
    )

    print(f"\nAgents agreeing: {count}/4")
    print("Records:", sample_count)
    print(f"Majority decision accuracy: {accuracy:.4f}")

    diagnostics.append({
        "analysis": "agreement_level",
        "category": f"{count}_of_4",
        "records": sample_count,
        "accuracy": accuracy,
        "actual_attack_rate": np.mean(group_true),
        "predicted_attack_rate": np.mean(group_pred),
        "false_positive_rate": np.mean(
            (group_true == 0) & (group_pred == 1)
        ) / max(np.mean(group_true == 0), 1e-12),
        "false_negative_rate": np.mean(
            (group_true == 1) & (group_pred == 0)
        ) / max(np.mean(group_true == 1), 1e-12)
    })

    # Class-specific accuracy
    for actual_class in [0, 1]:

        class_mask = group_true == actual_class

        if class_mask.sum() == 0:
            continue

        class_accuracy = accuracy_score(
            group_true[class_mask],
            group_pred[class_mask]
        )

        class_name = (
            "Normal" if actual_class == 0
            else "Attack"
        )

        print(
            f"{class_name} records: {class_mask.sum()}, "
            f"accuracy: {class_accuracy:.4f}"
        )


# --------------------------------------------------
# 5. IDENTIFY THE DISSENTING AGENT IN 3-1 SPLITS
# --------------------------------------------------

print("\nDISSENTING AGENT ANALYSIS")
print("=" * 65)

three_vote_mask = agreement_count == 3

dissenter_records = []

if three_vote_mask.any():

    group_indices = np.where(three_vote_mask)[0]

    for agent_index, agent in enumerate(agent_names):

        # Agent disagrees with the three-agent majority
        dissenter_mask = (
            prediction_matrix[group_indices, agent_index]
            != majority_prediction[group_indices]
        )

        selected_indices = group_indices[dissenter_mask]

        if len(selected_indices) == 0:
            continue

        dissenter_predictions = predictions[agent][selected_indices]

        dissenter_accuracy = accuracy_score(
            y_true[selected_indices],
            dissenter_predictions
        )

        majority_accuracy = accuracy_score(
            y_true[selected_indices],
            majority_prediction[selected_indices]
        )

        print(f"\n{agent}")
        print("Dissenter records:", len(selected_indices))
        print(f"Dissenter accuracy: {dissenter_accuracy:.4f}")
        print(f"Three-agent majority accuracy: {majority_accuracy:.4f}")

        dissenter_records.append({
            "analysis": "dissenter",
            "category": agent,
            "records": len(selected_indices),
            "accuracy": dissenter_accuracy,
            "actual_attack_rate": np.mean(
                y_true[selected_indices]
            ),
            "predicted_attack_rate": np.mean(
                dissenter_predictions
            ),
            "false_positive_rate": np.mean(
                (y_true[selected_indices] == 0)
                & (dissenter_predictions == 1)
            ) / max(
                np.mean(y_true[selected_indices] == 0),
                1e-12
            ),
            "false_negative_rate": np.mean(
                (y_true[selected_indices] == 1)
                & (dissenter_predictions == 0)
            ) / max(
                np.mean(y_true[selected_indices] == 1),
                1e-12
            )
        })

        diagnostics.append({
            "analysis": "majority_on_dissenter_records",
            "category": agent,
            "records": len(selected_indices),
            "accuracy": majority_accuracy,
            "actual_attack_rate": np.mean(
                y_true[selected_indices]
            ),
            "predicted_attack_rate": np.mean(
                majority_prediction[selected_indices]
            ),
            "false_positive_rate": np.mean(
                (y_true[selected_indices] == 0)
                & (majority_prediction[selected_indices] == 1)
            ) / max(
                np.mean(y_true[selected_indices] == 0),
                1e-12
            ),
            "false_negative_rate": np.mean(
                (y_true[selected_indices] == 1)
                & (majority_prediction[selected_indices] == 0)
            ) / max(
                np.mean(y_true[selected_indices] == 1),
                1e-12
            )
        })


# --------------------------------------------------
# 6. ANALYSE 2-2 TIES SEPARATELY
# --------------------------------------------------

print("\nTIE ANALYSIS: TWO NORMAL VS TWO ATTACK")
print("=" * 65)

tie_mask = agreement_count == 2

if tie_mask.any():

    tie_true = y_true[tie_mask]

    # Current system resolves ties as Normal
    tie_accuracy = accuracy_score(
        tie_true,
        np.zeros(len(tie_true), dtype=int)
    )

    print("Tie records:", int(tie_mask.sum()))
    print(f"Accuracy when ties resolve to Normal: {tie_accuracy:.4f}")
    print(f"Actual Normal rate: {np.mean(tie_true == 0):.4f}")
    print(f"Actual Attack rate: {np.mean(tie_true == 1):.4f}")

    diagnostics.append({
        "analysis": "tie_break_normal",
        "category": "2_vs_2",
        "records": int(tie_mask.sum()),
        "accuracy": tie_accuracy,
        "actual_attack_rate": np.mean(tie_true),
        "predicted_attack_rate": 0.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": np.mean(tie_true == 1)
    })


# --------------------------------------------------
# 7. UNANIMOUS VS SPLIT DECISIONS
# --------------------------------------------------

print("\nUNANIMOUS VS SPLIT DECISIONS")
print("=" * 65)

unanimous_mask = agreement_count == 4
split_mask = agreement_count < 4

for label, mask in [
    ("Unanimous decisions", unanimous_mask),
    ("Split decisions", split_mask)
]:

    count = int(mask.sum())

    if count == 0:
        continue

    accuracy = accuracy_score(
        y_true[mask],
        majority_prediction[mask]
    )

    print(f"\n{label}")
    print("Records:", count)
    print(f"Majority accuracy: {accuracy:.4f}")


# --------------------------------------------------
# 8. SAVE ALL ANALYSIS
# --------------------------------------------------

agreement_matrix.to_csv(
    f"{RESULTS_DIR}/agent_agreement_matrix.csv"
)

kappa_matrix.to_csv(
    f"{RESULTS_DIR}/agent_kappa_matrix.csv"
)

error_correlation.to_csv(
    f"{RESULTS_DIR}/agent_error_correlation.csv"
)

diagnostics_df = pd.DataFrame(diagnostics)

diagnostics_df.to_csv(
    f"{RESULTS_DIR}/agent_disagreement_diagnostics.csv",
    index=False
)

print("\nFILES SAVED")
print("1. results/agent_agreement_matrix.csv")
print("2. results/agent_kappa_matrix.csv")
print("3. results/agent_error_correlation.csv")
print("4. results/agent_disagreement_diagnostics.csv")

print("\nAGENT AGREEMENT ANALYSIS COMPLETED")
