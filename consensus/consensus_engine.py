
from collections import Counter
import math


class ConsensusEngine:

    def __init__(self, influence_graph):

        self.influence_graph = influence_graph

        self.static_weights = {
            "Traffic Agent": 0.90,
            "Threat Agent": 0.85,
            "Behaviour Agent": 0.70,
            "Risk Agent": 0.55
        }

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    def _validate_opinions(self, opinions):

        if not isinstance(opinions, dict):
            raise TypeError("Opinions must be a dictionary.")

        for agent, prediction in opinions.items():

            if prediction not in (0, 1):
                raise ValueError(
                    f"Invalid prediction from {agent}: {prediction}"
                )

            if agent not in self.influence_graph.graph:
                raise ValueError(
                    f"Agent '{agent}' is missing from influence graph."
                )

    # --------------------------------------------------
    # NORMALISE WEIGHTS
    # --------------------------------------------------

    def _normalise_weights(self, raw_weights):

        total = sum(raw_weights.values())

        if total <= 0:
            raise ValueError("Total weight must be positive.")

        return {
            agent: weight / total
            for agent, weight in raw_weights.items()
        }

    # --------------------------------------------------
    # GRAPH INFLUENCE FACTORS
    # --------------------------------------------------

    def _get_influence_factors(self, agents):

        graph = self.influence_graph.graph

        raw_influence = {}

        for agent in agents:

            outgoing_edges = graph.out_edges(
                agent,
                data=True
            )

            raw_influence[agent] = sum(
                data.get("influence", 0.0)
                for _, _, data in outgoing_edges
            )

        max_influence = max(
            raw_influence.values(),
            default=0.0
        )

        if max_influence <= 0:
            return {
                agent: 0.0
                for agent in agents
            }

        return {
            agent: raw_influence[agent] / max_influence
            for agent in agents
        }

    # --------------------------------------------------
    # METHOD 1: MAJORITY VOTING
    # --------------------------------------------------

    def majority_voting(self, opinions):

        if not opinions:
            return 0

        self._validate_opinions(opinions)

        votes = Counter(opinions.values())

        if votes[0] == votes[1]:
            return 0

        return votes.most_common(1)[0][0]

    # --------------------------------------------------
    # METHOD 2: STATIC WEIGHTED VOTING
    # --------------------------------------------------

    def static_weighted(self, opinions):

        if not opinions:
            return 0

        self._validate_opinions(opinions)

        scores = {0: 0.0, 1: 0.0}

        for agent, prediction in opinions.items():

            weight = self.static_weights.get(agent, 0.0)

            scores[prediction] += weight

        if math.isclose(scores[0], scores[1]):
            return 0

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # METHOD 3: ORIGINAL DYNAMIC WEIGHTS
    # --------------------------------------------------

    def calculate_dynamic_weights(self, opinions):

        if not opinions:
            return {}

        self._validate_opinions(opinions)

        graph = self.influence_graph.graph

        influence_factors = self._get_influence_factors(
            opinions.keys()
        )

        raw_weights = {}

        for agent in opinions:

            reliability = float(
                graph.nodes[agent]["reliability"]
            )

            influence = influence_factors[agent]

            raw_weights[agent] = reliability * (
                0.5 + 0.5 * influence
            )

        return self._normalise_weights(raw_weights)

    # --------------------------------------------------
    # METHOD 4: ORIGINAL DYNAMIC CONSENSUS
    # --------------------------------------------------

    def proposed_consensus(self, opinions):

        if not opinions:
            return 0

        self._validate_opinions(opinions)

        weights = self.calculate_dynamic_weights(opinions)

        scores = {0: 0.0, 1: 0.0}

        for agent, prediction in opinions.items():

            scores[prediction] += weights[agent]

        if math.isclose(scores[0], scores[1]):
            return 0

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # METHOD 5: CONTEXT-AWARE DYNAMIC WEIGHTS
    # --------------------------------------------------

    def calculate_contextual_weights(self, opinions):

        if not opinions:
            return {}

        self._validate_opinions(opinions)

        graph = self.influence_graph.graph

        influence_factors = self._get_influence_factors(
            opinions.keys()
        )

        raw_weights = {}

        for agent, prediction in opinions.items():

            # Reliability for the class this agent predicted
            reliability = float(
                self.influence_graph.get_prediction_reliability(
                    agent,
                    prediction
                )
            )

            if not 0 <= reliability <= 1:
                raise ValueError(
                    f"Invalid prediction reliability for {agent}"
                )

            influence = influence_factors[agent]

            # Combine prediction-specific reliability and influence
            raw_weights[agent] = reliability * (
                0.5 + 0.5 * influence
            )

        return self._normalise_weights(raw_weights)

    # --------------------------------------------------
    # METHOD 6: CONTEXT-AWARE CONSENSUS
    # --------------------------------------------------

    def context_aware_consensus(self, opinions):

        if not opinions:
            return 0

        weights = self.calculate_contextual_weights(opinions)

        scores = {0: 0.0, 1: 0.0}

        for agent, prediction in opinions.items():

            scores[prediction] += weights[agent]

        if math.isclose(scores[0], scores[1]):
            return 0

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # METHOD 7: CONFIDENCE-AWARE CONSENSUS
    # --------------------------------------------------

    def confidence_aware_consensus(self, probabilities):

        if not probabilities:
            return 0

        dynamic_weights = self.calculate_dynamic_weights(
            {agent: 0 for agent in probabilities}
        )

        scores = {0: 0.0, 1: 0.0}

        for agent, class_probabilities in probabilities.items():

            if len(class_probabilities) != 2:
                raise ValueError(
                    f"{agent} must provide two probabilities."
                )

            normal_probability = float(class_probabilities[0])
            attack_probability = float(class_probabilities[1])

            if not (
                math.isfinite(normal_probability)
                and math.isfinite(attack_probability)
            ):
                raise ValueError(
                    f"Invalid probabilities from {agent}."
                )

            if (
                normal_probability < 0
                or attack_probability < 0
                or normal_probability + attack_probability <= 0
            ):
                raise ValueError(
                    f"Invalid probability distribution from {agent}."
                )

            total = normal_probability + attack_probability

            normal_probability /= total
            attack_probability /= total

            weight = dynamic_weights[agent]

            scores[0] += weight * normal_probability
            scores[1] += weight * attack_probability

        if math.isclose(scores[0], scores[1]):
            return 0

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # METHOD 8: CONTEXT-AWARE CONFIDENCE CONSENSUS
    # --------------------------------------------------

    def context_aware_confidence_consensus(self, probabilities):

        if not probabilities:
            return 0

        opinions = {}

        for agent, class_probabilities in probabilities.items():

            if len(class_probabilities) != 2:
                raise ValueError(
                    f"{agent} must provide two probabilities."
                )

            normal_probability = float(class_probabilities[0])
            attack_probability = float(class_probabilities[1])

            if not (
                math.isfinite(normal_probability)
                and math.isfinite(attack_probability)
            ):
                raise ValueError(
                    f"Invalid probabilities from {agent}."
                )

            if (
                normal_probability < 0
                or attack_probability < 0
                or normal_probability + attack_probability <= 0
            ):
                raise ValueError(
                    f"Invalid probability distribution from {agent}."
                )

            opinions[agent] = (
                1 if attack_probability > normal_probability else 0
            )

        weights = self.calculate_contextual_weights(opinions)

        scores = {0: 0.0, 1: 0.0}

        for agent, class_probabilities in probabilities.items():

            normal_probability = float(class_probabilities[0])
            attack_probability = float(class_probabilities[1])

            total = normal_probability + attack_probability

            normal_probability /= total
            attack_probability /= total

            scores[0] += weights[agent] * normal_probability
            scores[1] += weights[agent] * attack_probability

        if math.isclose(scores[0], scores[1]):
            return 0

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # DISPLAY AND COMPARE
    # --------------------------------------------------

    def compare_methods(self, opinions):

        if not opinions:
            raise ValueError("No agent opinions provided.")

        self._validate_opinions(opinions)

        majority = self.majority_voting(opinions)
        static = self.static_weighted(opinions)
        proposed = self.proposed_consensus(opinions)
        contextual = self.context_aware_consensus(opinions)

        dynamic_weights = self.calculate_dynamic_weights(opinions)
        contextual_weights = self.calculate_contextual_weights(opinions)

        print("\nCONSENSUS COMPARISON")
        print("=" * 55)

        print("\nAGENT OPINIONS:")

        for agent, prediction in opinions.items():

            decision = "ATTACK" if prediction == 1 else "NORMAL"

            print(f"{agent}: {decision}")

        print("\nORIGINAL DYNAMIC WEIGHTS:")

        for agent, weight in dynamic_weights.items():
            print(f"{agent}: {weight:.4f}")

        print("\nCONTEXT-AWARE DYNAMIC WEIGHTS:")

        for agent, weight in contextual_weights.items():
            print(f"{agent}: {weight:.4f}")

        print("\nFINAL DECISIONS")

        print("Majority Voting MAS:", majority)
        print("Static Weighted MAS:", static)
        print("Original Dynamic IA-MAS:", proposed)
        print("Context-Aware IA-MAS:", contextual)

        return {
            "majority": majority,
            "static": static,
            "proposed": proposed,
            "context_aware": contextual,
            "dynamic_weights": dynamic_weights,
            "contextual_weights": contextual_weights
        }
