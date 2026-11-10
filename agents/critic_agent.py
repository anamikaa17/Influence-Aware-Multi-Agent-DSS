
class CriticAgent:

    def __init__(self):
        self.name = "Critic Agent"

    def evaluate(self, problem, opinions, evidence):

        print("\nCRITIC AGENT: EVALUATING DECISIONS")
        print("=" * 45)

        # Count the agent predictions
        malicious_votes = sum(opinions.values())
        normal_votes = len(opinions) - malicious_votes

        # Display disagreement
        print("\nAGENT DISAGREEMENT:")

        for agent, prediction in opinions.items():
            decision = "MALICIOUS" if prediction == 1 else "NORMAL"
            print(f"{agent}: {decision}")

        # Examine retrieved evidence
        print("\nEVIDENCE REVIEW:")

        for item in evidence:
            print(
                f"- {item['evidence']} "
                f"(Similarity: {item['similarity']:.4f})"
            )

        # Basic rule-based criticism
        print("\nCRITIC ANALYSIS:")

        if malicious_votes == normal_votes:
            status = "HIGH DISAGREEMENT"
            recommendation = "Further verification required."

        elif malicious_votes > normal_votes:
            status = "MAJORITY: MALICIOUS"

            if not evidence:
                recommendation = "Insufficient evidence. Verify before deciding."
            else:
                recommendation = (
                    "Malicious is the majority opinion. "
                    "Check evidence before final confirmation."
                )

        else:
            status = "MAJORITY: NORMAL"

            if not evidence:
                recommendation = "Insufficient evidence. Verify before deciding."
            else:
                recommendation = (
                    "Normal is the majority opinion. "
                    "Check evidence before final confirmation."
                )

        print("Status:", status)
        print("Recommendation:", recommendation)

        return {
            "status": status,
            "recommendation": recommendation,
            "malicious_votes": malicious_votes,
            "normal_votes": normal_votes
        }
