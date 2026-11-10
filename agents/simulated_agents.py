
import random
from dataclasses import dataclass


@dataclass
class SimulatedAgent:
    name: str
    accuracy: float
    seed: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def predict(self, actual_label):
        if self.rng.random() < self.accuracy:
            return actual_label

        return 1 - actual_label


def create_agents():
    return [
        SimulatedAgent("Traffic Agent", 0.90, 10),
        SimulatedAgent("Threat Agent", 0.85, 20),
        SimulatedAgent("Behaviour Agent", 0.70, 30),
        SimulatedAgent("Risk Agent", 0.55, 40)
    ]
