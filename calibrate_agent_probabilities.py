
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
MODEL_DIR = "models"
CALIBRATED_DIR = "models/calibrated"

os.makedirs(CALIBRATED_DIR, exist_ok=True)

print("\nLOADING TRAINING DATA")
df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label", "attack_cat", "id"])
y = df["label"]

# Separate data for calibration.
# The original model was trained on the full training dataset,
# so this calibration partition must not overlap with its training data.
X_model_train, X_calibration, y_model_train, y_calibration = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

trainer = MLAgentTrainer()

print("\nCALIBRATING SPECIALIST MODELS")
print("=" * 65)

calibration_results = {}

for agent, features in AGENT_FEATURES.items():

    print(f"\nProcessing {agent}...")

    model_path = os.path.join(
        MODEL_DIR,
        agent.lower().replace(" ", "_") + ".joblib"
    )

    # Load the original model
    original_model = joblib.load(model_path)

    # IMPORTANT:
    # This model was trained on the full official training set.
    # Its calibration data must be independent of that training data.
    # Therefore, we cannot use the original model with this split.
    # Instead, fit a fresh model on the training partition.

    base_model = trainer.prepare_model(features, df)

    base_model.fit(
        X_model_train[features],
        y_model_train
    )

    # Freeze the fitted model and learn only the calibration mapping
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid"
    )

    calibrated_model.fit(
        X_calibration[features],
        y_calibration
    )

    output_path = os.path.join(
        CALIBRATED_DIR,
        agent.lower().replace(" ", "_") + "_calibrated.joblib"
    )

    joblib.dump(calibrated_model, output_path)

    calibration_results[agent] = {
        "model_path": output_path,
        "calibration_method": "sigmoid"
    }

    print(f"{agent}: Calibration completed")
    print(f"Saved: {output_path}")

with open(
    os.path.join(CALIBRATED_DIR, "calibration_config.json"),
    "w"
) as file:
    json.dump(calibration_results, file, indent=4)

print("\nALL FOUR AGENTS CALIBRATED SUCCESSFULLY")
print("Original models remain unchanged.")
