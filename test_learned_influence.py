
import pandas as pd
import numpy as np

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer
from influence.influence_graph import DynamicInfluenceGraph
from sklearn.model_selection import train_test_split


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"

print("\nLOADING UNSW-NB15 DATA")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["label", "attack_cat", "id"])
y = df["label"].astype(int)

X_train, X_cal, y_train, y_cal = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

trainer = MLAgentTrainer()

models = {}
predictions = {}

print("\nTRAINING SPECIALIST AGENTS")

for agent, features in AGENT_FEATURES.items():

    model = trainer.prepare_model(features, df)

    model.fit(X_train[features], y_train)

    models[agent] = model

    predictions[agent] = model.predict(X_cal[features])

    print(f"{agent}: Completed")

print("\nLEARNING DATA-DRIVEN INFLUENCE")

graph = DynamicInfluenceGraph()

summary = graph.estimate_influence_from_predictions(
    y_cal.to_numpy(),
    predictions
)

print("\nAGENT SUMMARY")

for agent, values in summary.items():

    print(f"\n{agent}")
    print("Reliability:", round(values["reliability"], 4))
    print("Influence score:", round(values["influence_score"], 4))

graph.display_graph()

print("\nINFLUENCE LEARNING TEST COMPLETED")

import json
import os

os.makedirs("config", exist_ok=True)

learned_graph = {
    "nodes": {
        agent: graph.graph.nodes[agent]
        for agent in graph.graph.nodes
    },
    "edges": [
        {
            "source": source,
            "target": target,
            "base_weight": data.get("base_weight", 0),
            "information_gain": data.get("information_gain", 0),
            "influence": data.get("influence", 0)
        }
        for source, target, data in graph.graph.edges(data=True)
    ]
}

with open("config/learned_influence.json", "w") as file:
    json.dump(learned_graph, file, indent=4)

print("\nLearned influence graph saved successfully.")
