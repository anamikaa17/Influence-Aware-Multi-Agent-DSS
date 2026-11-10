
import pandas as pd

from agents.feature_groups import (
    AGENT_FEATURES,
    EXCLUDED_COLUMNS
)


def main():

    path = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"

    df = pd.read_csv(path)

    print("\nAGENT FEATURE VALIDATION")
    print("=" * 45)

    for agent, features in AGENT_FEATURES.items():

        missing = [
            feature for feature in features
            if feature not in df.columns
        ]

        leakage = [
            feature for feature in features
            if feature in EXCLUDED_COLUMNS
        ]

        print(f"\n{agent}")
        print("Number of features:", len(features))
        print("Missing columns:", missing)
        print("Target leakage:", leakage)

    print("\nVALIDATION COMPLETED")


if __name__ == "__main__":
    main()
