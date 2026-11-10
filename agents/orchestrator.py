
from agents.simulated_agents import create_agents
from agents.evidence_agent import EvidenceAgent
from agents.critic_agent import CriticAgent
from correction.self_correction import SelfCorrection


class Orchestrator:

    def __init__(self):
        self.agents = create_agents()
        self.correction_engine = SelfCorrection()

        self.evidence_agent = EvidenceAgent(
            "datasets/evidence/sample_evidence.txt"
        )

        self.critic = CriticAgent()

    def assign_task(self, problem):
        print("\nORCHESTRATOR: RECEIVED PROBLEM")
        print("Problem:", problem)

        for agent in self.agents:
            print(f"Task assigned to {agent.name}")

    def retrieve_evidence(self, problem):
        print("\nORCHESTRATOR: RETRIEVING EVIDENCE")

        evidence = self.evidence_agent.retrieve(
            problem,
            top_k=3
        )

        for item in evidence:
            print(
                f"\nEvidence: {item['evidence']}"
            )
            print(
                f"Similarity: {item['similarity']:.4f}"
            )

        return evidence

    def collect_opinions(self, actual_label):
        print("\nORCHESTRATOR: COLLECTING AGENT OPINIONS")

        opinions = {}

        for agent in self.agents:
            prediction = agent.predict(actual_label)
            opinions[agent.name] = prediction

            print(f"{agent.name} submitted: {prediction}")

        return opinions

    def execute(self, problem, actual_label):

        self.assign_task(problem)

        evidence = self.retrieve_evidence(problem)

        opinions = self.collect_opinions(actual_label)

        critic_result = self.critic.evaluate(
            problem,
            opinions,
            evidence
        )

        correction_result = self.correction_engine.correct(
            opinions,
            critic_result,
            evidence
        )

        return {
            "problem": problem,
            "evidence": evidence,
            "opinions": opinions,
            "critic_result": critic_result,
            "correction_result": correction_result
        }
        
    

