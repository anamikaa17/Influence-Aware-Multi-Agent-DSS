
import numpy as np
import json


class ConfidenceAwareInfluence:

    def __init__(
    self,
    graph,
    reliability_path="config/agent_reliability.json",
    confidence_path="config/confidence_reliability.json",
    learned_influence_path="config/learned_influence.json"
):

        self.graph = graph

        # Load validation-derived agent reliability
        with open(reliability_path, "r") as file:
            self.reliability_data = json.load(file)

        # Load confidence-bin reliability profiles
        with open(confidence_path, "r") as file:
            self.confidence_profiles = json.load(file)
            # Load data-driven learned influence graph
        with open(learned_influence_path, "r") as file:
             self.learned_influence_data = json.load(file)

    # --------------------------------------------------
    # 1. CONFIDENCE ADJUSTMENT
    # --------------------------------------------------

    def get_confidence_adjustment(self, agent_name, confidence):

        profile = self.confidence_profiles.get(agent_name, {})

        if not profile:
            return 0.5

        points = []

        for bin_name, values in profile.items():

            lower, upper = map(
                float,
                bin_name.split("-")
            )

            midpoint = (lower + upper) / 2

            adjustment = float(
                values["reliability_adjustment"]
            )

            points.append((midpoint, adjustment))

        points.sort(key=lambda item: item[0])

        midpoints = [
            point[0] for point in points
        ]

        adjustments = [
            point[1] for point in points
        ]

        interpolated = np.interp(
            confidence,
            midpoints,
            adjustments
        )

        return float(interpolated)

    # --------------------------------------------------
    # 2. GRAPH INFLUENCE FACTORS
    # --------------------------------------------------

    
            # --------------------------------------------------
    # 2. GRAPH INFLUENCE FACTORS
    # --------------------------------------------------

    def get_graph_factors(self, agent_names):

        outgoing_influence = {
            agent: 0.0
            for agent in agent_names
        }

        # Calculate influence using saved learned graph
        for edge in self.learned_influence_data["edges"]:

            source = edge["source"]

            if source in outgoing_influence:

                outgoing_influence[source] += float(
                    edge.get("influence", 0.0)
                )

        max_outgoing = max(
            outgoing_influence.values(),
            default=0
        )

        if max_outgoing <= 0:

            return {
                agent: 1.0
                for agent in agent_names
            }

        graph_factors = {
            agent: outgoing_influence[agent] / max_outgoing
            for agent in agent_names
        }

        return graph_factors

    # --------------------------------------------------
    # 3. ORIGINAL CONFIDENCE-AWARE WEIGHTS
    # --------------------------------------------------

    # Used by:
    # Advanced Interpolated IA-MAS

    def calculate_weights(self, probabilities):

        agent_names = list(probabilities.keys())

        graph_factors = self.get_graph_factors(
            agent_names
        )

        raw_weights = {}

        for agent_name, class_probabilities in probabilities.items():

            # Historical overall reliability
            historical_reliability = float(
                self.reliability_data[agent_name]["accuracy"]
            )

            # Model confidence
            confidence = float(
                max(class_probabilities)
            )

            # Confidence-bin adjustment
            confidence_factor = self.get_confidence_adjustment(
                agent_name,
                confidence
            )

            # Combine factors
            raw_weight = (
                historical_reliability
                * (0.5 + 0.5 * graph_factors[agent_name])
                * confidence_factor
            )

            raw_weights[agent_name] = raw_weight

        total_weight = sum(raw_weights.values())

        if total_weight <= 0:
            raise ValueError(
                "All calculated influence weights are zero."
            )

        normalized_weights = {
            agent: weight / total_weight
            for agent, weight in raw_weights.items()
        }

        return normalized_weights

    # --------------------------------------------------
    # 4. NEW CONTEXT-AWARE CONFIDENCE WEIGHTS
    # --------------------------------------------------

    # Used by:
    # Context-Aware Confidence IA-MAS (Method 7)

    def calculate_contextual_confidence_weights(self, probabilities):

        agent_names = list(probabilities.keys())

        graph_factors = self.get_graph_factors(
            agent_names
        )

        raw_weights = {}

        for agent_name, class_probabilities in probabilities.items():

            normal_probability = float(
                class_probabilities[0]
            )

            attack_probability = float(
                class_probabilities[1]
            )

            # Determine the agent's predicted class
            predicted_class = (
                1
                if attack_probability > normal_probability
                else 0
            )

            # Get prediction-specific historical precision
            node = self.graph.graph.nodes[agent_name]

            if predicted_class == 0:

                historical_reliability = float(
                    node["normal_precision"]
                )

            else:

                historical_reliability = float(
                    node["attack_precision"]
                )

            # Model confidence
            confidence = max(
                normal_probability,
                attack_probability
            )

            # Confidence-bin adjustment
            confidence_factor = self.get_confidence_adjustment(
                agent_name,
                confidence
            )

            # Combine contextual reliability,
            # graph influence and confidence adjustment
            raw_weight = (
                historical_reliability
                * (0.5 + 0.5 * graph_factors[agent_name])
                * confidence_factor
            )

            raw_weights[agent_name] = raw_weight

        total_weight = sum(raw_weights.values())

        if total_weight <= 0:
            raise ValueError(
                "All contextual confidence weights are zero."
            )

        normalized_weights = {
            agent: weight / total_weight
            for agent, weight in raw_weights.items()
        }

        return normalized_weights

    # --------------------------------------------------
    # 5. DISPLAY CONFIDENCE-AWARE WEIGHTS
    # --------------------------------------------------

    def display_weights(self, probabilities):

        weights = self.calculate_weights(
            probabilities
        )

        print("\nCONFIDENCE-AWARE DYNAMIC INFLUENCE")
        print("=" * 60)

        for agent, weight in weights.items():

            confidence = max(
                probabilities[agent]
            )

            adjustment = self.get_confidence_adjustment(
                agent,
                confidence
            )

            print(f"\n{agent}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Confidence adjustment: {adjustment:.4f}")
            print(f"Final influence weight: {weight:.4f}")

        print(
            "\nTotal normalized weight:",
            sum(weights.values())
        )

        return weights

    # --------------------------------------------------
    # 6. DISPLAY CONTEXT-AWARE CONFIDENCE WEIGHTS
    # --------------------------------------------------

    def display_contextual_weights(self, probabilities):

        weights = self.calculate_contextual_confidence_weights(
            probabilities
        )

        print("\nCONTEXT-AWARE CONFIDENCE INFLUENCE")
        print("=" * 60)

        for agent, weight in weights.items():

            normal_probability = probabilities[agent][0]
            attack_probability = probabilities[agent][1]

            predicted_class = (
                1
                if attack_probability > normal_probability
                else 0
            )

            node = self.graph.graph.nodes[agent]

            if predicted_class == 0:
                reliability = node["normal_precision"]
                class_name = "Normal"
            else:
                reliability = node["attack_precision"]
                class_name = "Attack"

            confidence = max(
                normal_probability,
                attack_probability
            )

            adjustment = self.get_confidence_adjustment(
                agent,
                confidence
            )

            print(f"\n{agent}")
            print(f"Predicted class: {class_name}")
            print(f"Prediction-specific precision: {reliability:.4f}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Confidence adjustment: {adjustment:.4f}")
            print(f"Final influence weight: {weight:.4f}")

        print(
            "\nTotal normalized weight:",
            sum(weights.values())
        )

        return weights
