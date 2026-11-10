
import pandas as pd

from agents.real_agent_runner import RealAgentRunner


TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

runner = RealAgentRunner()

for index in range(5):

    record = df.iloc[index]

    results = runner.predict_one_with_confidence(record)

    print(f"\nSAMPLE {index + 1}")

    for agent, result in results.items():

        prediction = (
            "ATTACK" if result["prediction"] == 1
            else "NORMAL"
        )

        confidence = result["confidence"] * 100

        print(
            f"{agent}: {prediction} "
            f"| Confidence: {confidence:.2f}%"
        )
