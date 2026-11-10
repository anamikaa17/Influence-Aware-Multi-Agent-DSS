
from agents.evidence_agent import EvidenceAgent


def main():

    evidence_agent = EvidenceAgent(
        "datasets/evidence/sample_evidence.txt"
    )

    query = "Detect suspicious repeated login attempts"

    results = evidence_agent.retrieve(query)

    print("\nEVIDENCE RETRIEVAL RESULTS")
    print("=" * 45)

    for item in results:

        print("\nEvidence:", item["evidence"])

        print(
            "Similarity:",
            round(item["similarity"], 4)
        )


if __name__ == "__main__":
    main()
