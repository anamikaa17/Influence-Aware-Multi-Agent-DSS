
import json
import pandas as pd

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph
from consensus.consensus_engine import ConsensusEngine


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

runner = RealAgentRunner()

graph = DynamicInfluenceGraph()

# Load validation reliability
with open("config/agent_reliability.json", "r") as file:
    reliability_data = json.load(file)

for agent, values in reliability_data.items():

    graph.graph.nodes[agent]["reliability"] = values["accuracy"]

graph.update_influence_weights()

engine = ConsensusEngine(graph)

print("\nCONFIDENCE-AWARE CONSENSUS TEST")
print("=" * 55)

for index in range(5):

    record = df.iloc[index]

    probabilities = {}

    for agent, features in AGENT_FEATURES.items():

        model = runner.models[agent]

        X = pd.DataFrame(
            [record[features].to_dict()],
            columns=features
        )

        probabilities[agent] = model.predict_proba(X)[0]

    decision = engine.confidence_aware_consensus(
        probabilities
    )

    actual = int(record["label"])

    print(f"\nSample {index + 1}")

    for agent, probs in probabilities.items():

        print(
            f"{agent}: "
            f"Normal={probs[0]:.4f}, "
            f"Attack={probs[1]:.4f}"
        )

    print(
        "Confidence-aware decision:",
        "ATTACK" if decision == 1 else "NORMAL"
    )

    print(
        "Actual:",
        "ATTACK" if actual == 1 else "NORMAL"
    )

    print(
        "Result:",
        "CORRECT" if decision == actual else "INCORRECT"
    )

print("\nTEST COMPLETED")
