
import os
import joblib
import pandas as pd

from agents.feature_groups import AGENT_FEATURES


MODEL_DIR = "models"


class RealAgentRunner:

    def __init__(self):

        self.models = {}

        print("\nLOADING SPECIALIST ML MODELS")

        for agent_name in AGENT_FEATURES:

            model_path = os.path.join(
                MODEL_DIR,
                agent_name.lower().replace(" ", "_") + ".joblib"
            )

            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)

            self.models[agent_name] = joblib.load(model_path)

            print(f"{agent_name}: Loaded successfully")

    def predict_one_with_confidence(self, record):

        results = {}

        for agent_name, features in AGENT_FEATURES.items():

            model = self.models[agent_name]

            input_data = pd.DataFrame(
                [record[features].to_dict()],
                columns=features
            )

            prediction = int(model.predict(input_data)[0])

            probabilities = model.predict_proba(input_data)[0]

            confidence = float(
                probabilities[list(model.classes_).index(prediction)]
            )

            results[agent_name] = {
                "prediction": prediction,
                "confidence": confidence
            }

        return results

    def predict_one(self, record):

        detailed_results = self.predict_one_with_confidence(record)

        return {
            agent: result["prediction"]
            for agent, result in detailed_results.items()
        }
