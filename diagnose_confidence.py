
import pandas as pd
import numpy as np

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

# Use 1000 records for diagnosis
sample = df.head(1000)

runner = RealAgentRunner()

print("\nCONFIDENCE VARIATION DIAGNOSTIC")
print("=" * 65)

for agent_name, features in AGENT_FEATURES.items():

    model = runner.models[agent_name]

    X = sample[features]

    probabilities = model.predict_proba(X)

    confidence = np.max(probabilities, axis=1)

    predictions = model.predict(X)

    print("\n", agent_name)
    print("-" * 45)

    print("Unique confidence values:",
          len(np.unique(np.round(confidence, 4))))

    print("Minimum confidence:",
          round(float(confidence.min()), 4))

    print("Maximum confidence:",
          round(float(confidence.max()), 4))

    print("Average confidence:",
          round(float(confidence.mean()), 4))

    print("Confidence standard deviation:",
          round(float(confidence.std()), 4))

    print("Unique predictions:",
          np.unique(predictions, return_counts=True))

    print("\nFirst 5 class probabilities:")

    for i in range(5):
        print(
            f"Sample {i + 1}: "
            f"Normal={probabilities[i][0]:.4f}, "
            f"Attack={probabilities[i][1]:.4f}"
        )

print("\nDIAGNOSTIC COMPLETED")
