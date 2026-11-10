
import pandas as pd

from sklearn.model_selection import train_test_split

from agents.evidence_agent import EvidenceAgent
from correction.self_correction import SelfCorrection


DATASET_PATH = (
    "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
)

EVIDENCE_PATH = (
    "datasets/evidence/sample_evidence.txt"
)

RANDOM_STATE = 42


def main():

    print("LOADING UNSW-NB15 DATASET")
    print("=" * 45)

    df = pd.read_csv(DATASET_PATH)

    train_df, remaining_df = train_test_split(
        df,
        test_size=0.40,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    print("Training records:", len(train_df))
    print("Remaining records:", len(remaining_df))

    # Initialize and train evidence reference.
    evidence_agent = EvidenceAgent(EVIDENCE_PATH)

    evidence_agent.fit_reference(train_df)

    # Select one record without using its label
    # for evidence generation.
    sample_record = remaining_df.iloc[0]

    print("\nGENERATING RECORD-SPECIFIC EVIDENCE")
    print("=" * 45)

    evidence = evidence_agent.generate_record_evidence(
        sample_record
    )

    print("Feature evidence generated:", len(evidence))

    # Aggregate feature evidence into specialist assessments.
    agent_evidence = evidence_agent.aggregate_agent_evidence(
        evidence
    )

    # Demonstration ML opinions.
    # These are deliberately simulated, not predictions
    # from the trained specialist ML models.
    opinions = {
        "Traffic Agent": 1,
        "Threat Agent": 0,
        "Behaviour Agent": 1,
        "Risk Agent": 1
    }

    critic_result = {
        "status": "MAJORITY: MALICIOUS",
        "recommendation": "Check evidence before final confirmation.",
        "malicious_votes": 3,
        "normal_votes": 1
    }

    # Run self-correction.
    correction_engine = SelfCorrection()

    result = correction_engine.correct(
        opinions=opinions,
        critic_result=critic_result,
        evidence=evidence,
        agent_evidence=agent_evidence
    )

    print("\nFINAL INTEGRATED OUTPUT")
    print("=" * 45)

    print("Original opinions:", result["original_opinions"])
    print("Corrected opinions:", result["corrected_opinions"])
    print("Majority decision:", result["majority_decision"])
    print("Evidence decision:", result["evidence_decision"])
    print("Final decision:", result["final_decision"])

    print("\nEvaluation label is intentionally not used "
          "for generating the decision.")


if __name__ == "__main__":
    main()
