
import numpy as np
import pandas as pd

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES
from consensus.consensus_engine import ConsensusEngine
from influence.influence_graph import DynamicInfluenceGraph


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

runner = RealAgentRunner()
graph = DynamicInfluenceGraph()
engine = ConsensusEngine(graph)

# Generate predictions for all agents
agent_predictions = {}

for agent, features in AGENT_FEATURES.items():

    agent_predictions[agent] = runner.models[agent].predict(
        df[features]
    )

# Compare consensus outputs
static_results = []
proposed_results = []

for i in range(len(df)):

    opinions = {
        agent: int(predictions[i])
        for agent, predictions in agent_predictions.items()
    }

    static_results.append(
        engine.static_weighted(opinions)
    )

    proposed_results.append(
        engine.proposed_consensus(opinions)
    )

static_results = np.array(static_results)
proposed_results = np.array(proposed_results)

# Compare weights
dynamic_weights = engine.calculate_dynamic_weights({
    agent: 0 for agent in AGENT_FEATURES
})

print("\nDYNAMIC WEIGHTS")
print("=" * 45)

for agent, weight in dynamic_weights.items():
    print(f"{agent}: {weight:.4f}")

print("\nDECISION COMPARISON")
print("=" * 45)

print("Total records:", len(df))

print(
    "Static and proposed decisions identical:",
    np.array_equal(static_results, proposed_results)
)

print(
    "Number of different decisions:",
    np.sum(static_results != proposed_results)
)

print(
    "Static Attack decisions:",
    np.sum(static_results == 1)
)

print(
    "Proposed Attack decisions:",
    np.sum(proposed_results == 1)
)

print("\nDIAGNOSTIC COMPLETED")
