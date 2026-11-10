import pandas as pd
import json
import os

INPUT_PATH = "results/agent_confidence_bins.csv"
OUTPUT_PATH = "config/self_correction_reliability.json"


def main():

    df = pd.read_csv(INPUT_PATH)

    reliability = {}

    for agent in df["Agent"].unique():

        agent_df = df[df["Agent"] == agent]

        bins = {}

        for _, row in agent_df.iterrows():

            confidence_range = row["Confidence Range"]

            bins[confidence_range] = {
                "records": int(row["Records"]),
                "accuracy": float(row["Accuracy"]),
                "mean_confidence": float(
                    row["Mean Confidence"]
                )
            }

        reliability[agent] = {
            "confidence_bins": bins
        }

    os.makedirs("config", exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:

        json.dump(
            reliability,
            file,
            indent=4
        )

    print("\nCONFIDENCE RELIABILITY PROFILE CREATED")

    for agent, data in reliability.items():

        print("\n", agent)

        for bin_name, values in data[
            "confidence_bins"
        ].items():

            print(
                bin_name,
                "Accuracy:",
                round(values["accuracy"], 4),
                "Records:",
                values["records"]
            )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()