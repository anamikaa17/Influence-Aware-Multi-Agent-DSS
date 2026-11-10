
import pandas as pd

from sklearn.model_selection import train_test_split

from agents.evidence_agent import EvidenceAgent


DATASET_PATH = (
    "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
)

EVIDENCE_PATH = (
    "datasets/evidence/sample_evidence.txt"
)

RANDOM_STATE = 42


def main():

    print("Loading UNSW-NB15 training data...")

    df = pd.read_csv(DATASET_PATH)

    # Reproduce the existing 60/20/20 experiment split.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.40,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    print("Training records:", len(train_df))
    print("Remaining records:", len(remaining_df))

    # Initialize evidence agent.
    evidence_agent = EvidenceAgent(EVIDENCE_PATH)

    # Learn statistical reference distributions
    # using only the training partition.
    evidence_agent.fit_reference(
        train_df,
        target_column="label"
    )

    # Select one record from the remaining partition.
    sample_record = remaining_df.iloc[0]

    print("\nSAMPLE RECORD")
    print("=" * 45)
    print("Record index:", sample_record.name)

    # Generate feature-level evidence.
    evidence = evidence_agent.generate_record_evidence(
        sample_record
    )

    print("\nRECORD-SPECIFIC EVIDENCE")
    print("=" * 45)

    if not evidence:
        print("No sufficiently strong feature evidence.")

    for item in evidence:

        print(f"\nAgent: {item['agent']}")
        print("Feature:", item["feature"])
        print("Value:", item["value"])
        print("Direction:", item["direction"])
        print(
            "Evidence strength:",
            round(item["evidence_strength"], 3)
        )
        print("Explanation:", item["evidence"])

    print("\nTotal evidence items:", len(evidence))

    # Aggregate feature observations into
    # four specialist-level assessments.
    print("\nAGENT-LEVEL EVIDENCE AGGREGATION")
    print("=" * 45)

    agent_results = evidence_agent.aggregate_agent_evidence(
        evidence
    )

    for agent_name, result in agent_results.items():

        decision = result["decision"]

        if decision == 1:
            decision_text = "ATTACK"
        elif decision == 0:
            decision_text = "NORMAL"
        else:
            decision_text = "UNCERTAIN"

        print(f"\n{agent_name}")
        print("Decision:", decision_text)
        print(
            "Attack strength:",
            result["attack_strength"]
        )
        print(
            "Normal strength:",
            result["normal_strength"]
        )
        print(
            "Attack features:",
            result["attack_features"]
        )
        print(
            "Normal features:",
            result["normal_features"]
        )
        print(
            "Total features:",
            result["total_features"]
        )

    print("\nEVIDENCE GENERATION TEST COMPLETED")


if __name__ == "__main__":
    main()
