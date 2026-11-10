
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from agents.feature_groups import AGENT_FEATURES
from agents.real_agent_runner import RealAgentRunner


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
CALIBRATED_DIR = "models/calibrated"

df = pd.read_csv(TEST_PATH)
y_true = df["label"].astype(int).to_numpy()

# Load original models
original_runner = RealAgentRunner()

print("\nORIGINAL VS CALIBRATED MODEL COMPARISON")
print("=" * 75)

results = []

for agent, features in AGENT_FEATURES.items():

    print(f"\n{agent}")
    print("-" * 55)

    X = df[features]

    # Original model probabilities
    original_model = original_runner.models[agent]

    original_probabilities = original_model.predict_proba(X)
    original_classes = list(original_model.classes_)
    original_attack_index = original_classes.index(1)

    original_attack_prob = original_probabilities[:, original_attack_index]

    # Load calibrated model
    calibrated_path = os.path.join(
        CALIBRATED_DIR,
        agent.lower().replace(" ", "_") + "_calibrated.joblib"
    )

    calibrated_model = joblib.load(calibrated_path)

    calibrated_probabilities = calibrated_model.predict_proba(X)
    calibrated_classes = list(calibrated_model.classes_)
    calibrated_attack_index = calibrated_classes.index(1)

    calibrated_attack_prob = calibrated_probabilities[:, calibrated_attack_index]

    # Calculate Brier scores
    original_brier = brier_score_loss(
        y_true,
        original_attack_prob
    )

    calibrated_brier = brier_score_loss(
        y_true,
        calibrated_attack_prob
    )

    # ECE calculation using confidence of predicted class
    def calculate_ece(model_probabilities):

        predicted_classes = np.argmax(
            model_probabilities,
            axis=1
        )

        confidence = np.max(
            model_probabilities,
            axis=1
        )

        correct = (
            predicted_classes == y_true
        ).astype(int)

        edges = np.linspace(0, 1, 11)
        ece = 0.0

        for i in range(10):

            if i == 9:
                mask = (
                    (confidence >= edges[i]) &
                    (confidence <= edges[i + 1])
                )
            else:
                mask = (
                    (confidence >= edges[i]) &
                    (confidence < edges[i + 1])
                )

            if np.any(mask):

                average_confidence = confidence[mask].mean()
                accuracy = correct[mask].mean()

                ece += (
                    np.sum(mask) / len(confidence)
                ) * abs(average_confidence - accuracy)

        return ece

    original_ece = calculate_ece(original_probabilities)
    calibrated_ece = calculate_ece(calibrated_probabilities)

    results.append({
        "Agent": agent,
        "Original Brier": original_brier,
        "Calibrated Brier": calibrated_brier,
        "Original ECE": original_ece,
        "Calibrated ECE": calibrated_ece
    })

    print(f"Original Brier:   {original_brier:.4f}")
    print(f"Calibrated Brier: {calibrated_brier:.4f}")
    print(f"Original ECE:     {original_ece:.4f}")
    print(f"Calibrated ECE:   {calibrated_ece:.4f}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/calibrated_model_comparison.csv",
    index=False
)

print("\nFINAL COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False))

print("\nSaved to results/calibrated_model_comparison.csv")
print("\nCOMPARISON COMPLETED")
