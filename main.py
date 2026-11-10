
import random
from collections import Counter

from agents.simulated_agents import create_agents
from influence.influence_graph import DynamicInfluenceGraph
from consensus.consensus_engine import ConsensusEngine
from agents.orchestrator import Orchestrator


def majority_voting(opinions):
    votes = Counter(opinions.values())
    return votes.most_common(1)[0][0]


def run_experiment():

    agents = create_agents()

    influence_graph = DynamicInfluenceGraph()

    engine = ConsensusEngine(influence_graph)
    orchestrator = Orchestrator()

    total_samples = 1000
    correct_predictions = 0
    rng = random.Random(42)

    print("\nINFLUENCE-AWARE MULTI-AGENT DSS")
    print("=" * 45)

    # Majority Voting Experiment
    for _ in range(total_samples):

        actual_label = rng.randint(0, 1)

        opinions = {}

        for agent in agents:
            opinions[agent.name] = agent.predict(actual_label)

        final_decision = majority_voting(opinions)

        if final_decision == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples * 100

    print("\nAGENTS CREATED:")

    for agent in agents:
        print(f"{agent.name}: {agent.accuracy * 100:.0f}%")

    print("\nEXPERIMENT RESULTS")
    print("Total samples:", total_samples)
    print("Correct decisions:", correct_predictions)
    print(f"Majority voting accuracy: {accuracy:.2f}%")

    # Initial Influence Graph
    print("\nINITIAL INFLUENCE GRAPH")

    influence_graph.display_graph()

    # Reliability Update
    print("\nUPDATING THREAT AGENT RELIABILITY")

    influence_graph.update_reliability(
        "Threat Agent",
        correct=60,
        total=100
    )

    influence_graph.display_graph()

    # Consensus Engine Test
    print("\nTESTING CONSENSUS ENGINE")

    test_opinions = {
        "Traffic Agent": 1,
        "Threat Agent": 1,
        "Behaviour Agent": 0,
        "Risk Agent": 0
    }

    engine.compare_methods(test_opinions)
    print("\nTESTING ORCHESTRATOR")

    test_problem = "Determine whether the network traffic is malicious."

    orchestrator_opinions = orchestrator.execute(
    test_problem,
    actual_label=1
)

    print("\nORCHESTRATOR OUTPUT:")
    print(orchestrator_opinions)


if __name__ == "__main__":
    run_experiment()
