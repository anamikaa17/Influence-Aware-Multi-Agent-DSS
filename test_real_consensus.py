
import pandas as pd

from agents.real_agent_runner import RealAgentRunner
from influence.influence_graph import DynamicInfluenceGraph
from consensus.consensus_engine import ConsensusEngine


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"


# Load test dataset
df = pd.read_csv(TEST_PATH)

# Load real specialist agents
runner = RealAgentRunner()

# Initialise influence graph and consensus engine
influence_graph = DynamicInfluenceGraph()
consensus_engine = ConsensusEngine(influence_graph)

print("\nREAL MULTI-AGENT CONSENSUS TEST")
print("=" * 60)

# Test first five records
for index in range(5):

    record = df.iloc[index]

    # Get predictions from all four real agents
    opinions = runner.predict_one(record)

    print(f"\n\nSAMPLE {index + 1}")
    print("-" * 45)

    # Actual label is used only for evaluation, never prediction
    actual_label = int(record["label"])

    print(
        "Actual:",
        "ATTACK" if actual_label == 1 else "NORMAL"
    )

    # Compare the three consensus methods
    results = consensus_engine.compare_methods(opinions)

    print("\nDECISION CHECK")

    for method, decision in results.items():

        status = "ATTACK" if decision == 1 else "NORMAL"

        correctness = (
            "CORRECT" if decision == actual_label
            else "INCORRECT"
        )

        print(f"{method}: {status} - {correctness}")


print("\nREAL MULTI-AGENT TEST COMPLETED")
