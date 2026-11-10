
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score
)
from sklearn.calibration import calibration_curve

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"

RESULT_PATH = "results/agent_calibration_results.csv"

os.makedirs("results", exist_ok=True)

print("\nLOADING TRAINING DATA")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(
    columns=["label", "attack_cat", "id"]
)

y = df["label"].astype(int)

# --------------------------------------------------
# 1. STRATIFIED TRAINING / VALIDATION SPLIT
# --------------------------------------------------

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_true = y_validation.to_numpy()

print("\nTRAINING RECORDS:", len(X_train))
print("VALIDATION RECORDS:", len(X_validation))

trainer = MLAgentTrainer()

results = []

print("\nAGENT PROBABILITY CALIBRATION EVALUATION")
print("=" * 70)


# --------------------------------------------------
# 2. EVALUATE EACH SPECIALIST AGENT
# --------------------------------------------------

for agent, features in AGENT_FEATURES.items():

    print(f"\nProcessing {agent}")

    # Train a temporary model using training partition only
    model = trainer.prepare_model(features, df)

    model.fit(
        X_train[features],
        y_train
    )

    # Predict validation probabilities
    probabilities = model.predict_proba(
        X_validation[features]
    )

    class_indices = list(model.classes_)

    normal_index = class_indices.index(0)
    attack_index = class_indices.index(1)

    attack_probabilities = probabilities[:, attack_index]

    normal_probabilities = probabilities[:, normal_index]

    # Predicted class and top-label confidence
    predictions = model.predict(
        X_validation[features]
    )

    confidence = np.max(probabilities, axis=1)

    correct = (
        predictions == y_true
    ).astype(int)

    # --------------------------------------------------
    # 3. BRIER SCORE
    # --------------------------------------------------

    brier = brier_score_loss(
        y_true,
        attack_probabilities
    )

    # --------------------------------------------------
    # 4. TOP-LABEL EXPECTED CALIBRATION ERROR
    # --------------------------------------------------

    bin_edges = np.linspace(0, 1, 11)

    ece = 0.0

    for i in range(10):

        if i == 9:

            mask = (
                (confidence >= bin_edges[i]) &
                (confidence <= bin_edges[i + 1])
            )

        else:

            mask = (
                (confidence >= bin_edges[i]) &
                (confidence < bin_edges[i + 1])
            )

        if np.any(mask):

            average_confidence = confidence[mask].mean()

            observed_accuracy = correct[mask].mean()

            ece += (
                np.sum(mask) / len(confidence)
            ) * abs(
                average_confidence - observed_accuracy
            )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    # --------------------------------------------------
    # 5. SAVE METRICS
    # --------------------------------------------------

    results.append({

        "Agent": agent,

        "Accuracy": accuracy,

        "Brier Score": brier,

        "Top Label ECE": ece,

        "Validation Samples": len(y_true)

    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Top Label ECE: {ece:.4f}")

    # --------------------------------------------------
    # 6. ATTACK PROBABILITY CALIBRATION CURVE
    # --------------------------------------------------

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        attack_probabilities,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=agent
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel("Mean predicted attack probability")

    plt.ylabel("Observed attack frequency")

    plt.title(
        f"Calibration Curve - {agent}"
    )

    plt.legend()

    plt.grid(True)

    filename = (
        agent.lower().replace(" ", "_")
        + "_calibration.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Calibration graph saved: {filename}")


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nCALIBRATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print(f"\nResults saved to: {RESULT_PATH}")

print("\nVALIDATION CALIBRATION EVALUATION COMPLETED")
