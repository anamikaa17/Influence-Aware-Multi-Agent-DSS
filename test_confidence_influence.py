
import pandas as pd

from agents.real_agent_runner import RealAgentRunner
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph
from influence.confidence_aware_influence import ConfidenceAwareInfluence


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

runner = RealAgentRunner()
graph = DynamicInfluenceGraph()

influence_engine = ConfidenceAwareInfluence(graph)

print("\nTESTING CONFIDENCE-AWARE INFLUENCE")
print("=" * 60)

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

    print(f"\nSAMPLE {index + 1}")

    influence_engine.display_weights(probabilities)

print("\nTEST COMPLETED")
