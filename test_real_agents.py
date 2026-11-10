
import joblib
import pandas as pd

from agents.feature_groups import AGENT_FEATURES

TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

MODEL_DIR = "models"

# Load test data
df = pd.read_csv(TEST_PATH)

# Select five samples for a quick prediction test
sample = df.head(5)

print("\nTESTING REAL SPECIALIST AGENTS")
print("=" * 60)

for agent_name, features in AGENT_FEATURES.items():

    model_path = (
        MODEL_DIR
        + "/"
        + agent_name.lower().replace(" ", "_")
        + ".joblib"
    )

    model = joblib.load(model_path)

    X = sample[features]

    predictions = model.predict(X)

    print("\n", agent_name)
    print("-" * 40)

    for i, prediction in enumerate(predictions):

        result = "ATTACK" if prediction == 1 else "NORMAL"

        print(f"Sample {i + 1}: {result}")

print("\nALL REAL AGENTS TESTED SUCCESSFULLY")
