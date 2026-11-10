
import numpy as np
import pandas as pd

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)
y_true = df["label"].astype(int).to_numpy()

runner = RealAgentRunner()

confidence_bins = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

all_results = []

print("\nAGENT CONFIDENCE BIN ANALYSIS")
print("=" * 75)

for agent, features in AGENT_FEATURES.items():

    print(f"\n{agent}")
    print("-" * 75)

    model = runner.models[agent]

    probabilities = model.predict_proba(df[features])
    predictions = model.predict(df[features])

    confidence = np.max(probabilities, axis=1)
    correct = (predictions == y_true).astype(int)

    for lower, upper in confidence_bins:

        mask = (
            (confidence >= lower) &
            (confidence < upper)
        )

        count = int(np.sum(mask))

        if count == 0:
            continue

        average_confidence = float(
            confidence[mask].mean()
        )

        actual_accuracy = float(
            correct[mask].mean()
        )

        gap = abs(
            average_confidence - actual_accuracy
        )

        if average_confidence > actual_accuracy:
            status = "OVERCONFIDENT"
        elif average_confidence < actual_accuracy:
            status = "UNDERCONFIDENT"
        else:
            status = "WELL CALIBRATED"

        result = {
            "Agent": agent,
            "Confidence Range": f"{lower:.0%}-{min(upper, 1.0):.0%}",
            "Samples": count,
            "Average Confidence": average_confidence,
            "Actual Accuracy": actual_accuracy,
            "Calibration Gap": gap,
            "Status": status
        }

        all_results.append(result)

        print(
            f"{lower:.0%}-{min(upper, 1.0):.0%} | "
            f"Samples: {count:6d} | "
            f"Confidence: {average_confidence:.4f} | "
            f"Accuracy: {actual_accuracy:.4f} | "
            f"Gap: {gap:.4f} | "
            f"{status}"
        )

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "results/confidence_bin_analysis.csv",
    index=False
)

print("\nRESULTS SAVED")
print("results/confidence_bin_analysis.csv")

print("\nANALYSIS COMPLETED")
