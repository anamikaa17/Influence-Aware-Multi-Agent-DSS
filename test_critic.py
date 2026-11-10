
from agents.critic_agent import CriticAgent


def main():

    critic = CriticAgent()

    opinions = {
        "Traffic Agent": 1,
        "Threat Agent": 0,
        "Behaviour Agent": 1,
        "Risk Agent": 1
    }

    evidence = [
        {
            "evidence": "A sudden increase in traffic volume may indicate suspicious network behaviour.",
            "similarity": 0.6113
        },
        {
            "evidence": "Unexpected communication with unknown external IP addresses may indicate malicious activity.",
            "similarity": 0.5610
        }
    ]

    result = critic.evaluate(
        "Determine whether network traffic is malicious.",
        opinions,
        evidence
    )

    print("\nCRITIC OUTPUT:")
    print(result)


if __name__ == "__main__":
    main()
