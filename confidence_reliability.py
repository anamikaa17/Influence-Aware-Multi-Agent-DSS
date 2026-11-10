
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
OUTPUT_PATH = "config/confidence_reliability.json"

CONFIDENCE_BINS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

# Prior strength for sample-size-aware smoothing
SMOOTHING_STRENGTH = 200

os.makedirs("config", exist_ok=True)

print("\nLOADING TRAINING DATA")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(
    columns=["label", "attack_cat", "id"]
)

y = df["label"].astype(int)

# Independent training and validation partitions
X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

trainer = MLAgentTrainer()

confidence_profiles = {}

print("\nBUILDING IMPROVED CONFIDENCE PROFILES")
print("=" * 75)


for agent, features in AGENT_FEATURES.items():

    print(f"\nProcessing {agent}")

    # Train specialist on training partition
    model = trainer.prepare_model(features, df)

    model.fit(
        X_train[features],
        y_train
    )

    # Validation probabilities
    probabilities = model.predict_proba(
        X_validation[features]
    )

    classes = list(model.classes_)

    # Identify probability columns correctly
    class_indices = [
        classes.index(0),
        classes.index(1)
    ]

    probabilities = probabilities[:, class_indices]

    predictions = np.array(classes)[
        np.argmax(probabilities, axis=1)
    ]

    confidence = np.max(probabilities, axis=1)

    correct = (
        predictions == y_validation.to_numpy()
    ).astype(int)

    # Overall validation accuracy becomes the prior
    prior_accuracy = float(correct.mean())

    agent_profile = {}

    print("-" * 75)

    print(
        f"Overall validation accuracy: {prior_accuracy:.4f}"
    )

    for lower, upper in CONFIDENCE_BINS:

        mask = (
            (confidence >= lower) &
            (confidence < upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        average_confidence = float(
            confidence[mask].mean()
        )

        actual_accuracy = float(
            correct[mask].mean()
        )

        # --------------------------------------------------
        # 1. SAMPLE-SIZE-AWARE ACCURACY SMOOTHING
        # --------------------------------------------------

        smoothed_accuracy = (
            count * actual_accuracy
            + SMOOTHING_STRENGTH * prior_accuracy
        ) / (
            count + SMOOTHING_STRENGTH
        )

        # --------------------------------------------------
        # 2. CALIBRATION GAP
        # --------------------------------------------------

        raw_gap = (
            average_confidence - actual_accuracy
        )

        # Shrink calibration gap toward zero when
        # the confidence bin has fewer observations.
        shrinkage_factor = (
            count / (count + SMOOTHING_STRENGTH)
        )

        smoothed_gap = (
            raw_gap * shrinkage_factor
        )

        # --------------------------------------------------
        # 3. CONFIDENCE RELIABILITY ADJUSTMENT
        # --------------------------------------------------

        # Penalize overconfidence.
        # Underconfidence does not receive an artificial penalty.
        reliability_adjustment = float(
            np.clip(
                1.0 - max(0.0, smoothed_gap),
                0.0,
                1.0
            )
        )

        bin_name = (
            f"{lower:.2f}-{min(upper, 1.0):.2f}"
        )

        agent_profile[bin_name] = {

            "samples": count,

            "average_confidence": average_confidence,

            "actual_accuracy": actual_accuracy,

            "smoothed_accuracy": float(
                smoothed_accuracy
            ),

            "prior_accuracy": prior_accuracy,

            "calibration_gap": float(raw_gap),

            "smoothed_calibration_gap": float(
                smoothed_gap
            ),

            "shrinkage_factor": float(
                shrinkage_factor
            ),

            "reliability_adjustment": reliability_adjustment
        }

        print(
            f"{bin_name} | "
            f"Samples: {count:6d} | "
            f"Confidence: {average_confidence:.4f} | "
            f"Accuracy: {actual_accuracy:.4f} | "
            f"Smoothed accuracy: {smoothed_accuracy:.4f} | "
            f"Adjustment: {reliability_adjustment:.4f}"
        )

    confidence_profiles[agent] = agent_profile


# --------------------------------------------------
# 4. SAVE CONFIDENCE PROFILES
# --------------------------------------------------

with open(OUTPUT_PATH, "w") as file:

    json.dump(
        confidence_profiles,
        file,
        indent=4
    )

print("\nIMPROVED CONFIDENCE PROFILES CREATED")

print(f"Saved to: {OUTPUT_PATH}")

print("\nSMOOTHING STRENGTH:", SMOOTHING_STRENGTH)

print("\nCONFIDENCE RELIABILITY GENERATION COMPLETED")
