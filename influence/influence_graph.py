
import networkx as nx
import numpy as np


class DynamicInfluenceGraph:

    def __init__(self):

        self.graph = nx.DiGraph()

        self.agents = {
            "Traffic Agent": 0.90,
            "Threat Agent": 0.85,
            "Behaviour Agent": 0.70,
            "Risk Agent": 0.55
        }

        self.build_graph()

    # --------------------------------------------------
    # 1. BUILD INITIAL GRAPH
    # --------------------------------------------------

    def build_graph(self):

        for name, reliability in self.agents.items():

            self.graph.add_node(
                name,
                reliability=reliability,

                normal_reliability=reliability,
                attack_reliability=reliability,

                normal_precision=reliability,
                attack_precision=reliability,

                normal_samples=0,
                attack_samples=0,

                influence_score=0.0
            )

        # Initially create all possible directed relationships.
        # Learned calibration data will determine their strengths.

        agent_names = list(self.agents.keys())

        for source in agent_names:
            for target in agent_names:

                if source == target:
                    continue

                self.graph.add_edge(
                    source,
                    target,
                    base_weight=0.0,
                    influence=0.0,
                    information_gain=0.0
                )

        self.update_influence_weights()

    # --------------------------------------------------
    # 2. UPDATE INFLUENCE WEIGHTS
    # --------------------------------------------------

    def update_influence_weights(self):

        for target in self.graph.nodes:

            incoming = list(
                self.graph.in_edges(target, data=True)
            )

            total = sum(
                data["base_weight"]
                * self.graph.nodes[source]["reliability"]
                for source, _, data in incoming
            )

            for source, _, data in incoming:

                score = (
                    data["base_weight"]
                    * self.graph.nodes[source]["reliability"]
                )

                data["influence"] = (
                    score / total if total > 0 else 0.0
                )

        self.update_agent_influence_scores()

    # --------------------------------------------------
    # 3. CALCULATE AGENT INFLUENCE SCORES
    # --------------------------------------------------

    def update_agent_influence_scores(self):

        for agent in self.graph.nodes:

            outgoing = list(
                self.graph.out_edges(agent, data=True)
            )

            score = sum(
                data["influence"]
                for _, _, data in outgoing
            )

            self.graph.nodes[agent][
                "influence_score"
            ] = float(score)

    # --------------------------------------------------
    # 4. CONDITIONAL LOG-LOSS
    # --------------------------------------------------

    @staticmethod
    def _conditional_log_loss(
        y_true,
        target_predictions,
        source_predictions=None
    ):

        y_true = np.asarray(y_true, dtype=int)
        target_predictions = np.asarray(
            target_predictions, dtype=int
        )

        if source_predictions is not None:
            source_predictions = np.asarray(
                source_predictions, dtype=int
            )

        n = len(y_true)

        losses = []

        for i in range(n):

            # Leave-one-out estimation:
            # The current sample is excluded from its own
            # conditional probability estimate.

            mask = np.ones(n, dtype=bool)
            mask[i] = False

            target_mask = (
                target_predictions[mask]
                == target_predictions[i]
            )

            if source_predictions is not None:

                source_mask = (
                    source_predictions[mask]
                    == source_predictions[i]
                )

                mask_indices = np.flatnonzero(mask)

                selected_indices = mask_indices[
                    target_mask & source_mask
                ]

            else:

                mask_indices = np.flatnonzero(mask)

                selected_indices = mask_indices[
                    target_mask
                ]

            if len(selected_indices) == 0:

                probability = 0.5

            else:

                matching_labels = y_true[selected_indices]

                # Laplace smoothing
                count_correct_class = int(
                    np.sum(matching_labels == y_true[i])
                )

                probability = (
                    count_correct_class + 1
                ) / (
                    len(selected_indices) + 2
                )

            probability = np.clip(
                probability,
                1e-10,
                1.0
            )

            losses.append(-np.log(probability))

        return float(np.mean(losses))

    # --------------------------------------------------
    # 5. LEARN DATA-DRIVEN INFLUENCE
    # --------------------------------------------------

    def estimate_influence_from_predictions(
        self,
        y_true,
        agent_predictions
    ):

        y_true = np.asarray(y_true, dtype=int)

        if len(y_true) == 0:
            raise ValueError(
                "Calibration data cannot be empty."
            )

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if not isinstance(agent_predictions, dict):
            raise TypeError(
                "Agent predictions must be a dictionary."
            )

        if set(agent_predictions.keys()) != set(self.graph.nodes):
            raise ValueError(
                "Predictions must be supplied for every graph agent."
            )

        predictions = {}

        for agent, values in agent_predictions.items():

            values = np.asarray(values, dtype=int)

            if len(values) != len(y_true):
                raise ValueError(
                    f"Prediction length mismatch for {agent}."
                )

            if not np.isin(values, [0, 1]).all():
                raise ValueError(
                    f"Invalid predictions from {agent}."
                )

            predictions[agent] = values

        # First estimate individual agent reliability.

        for agent, pred in predictions.items():

            correct = int(np.sum(pred == y_true))
            total = len(y_true)

            node = self.graph.nodes[agent]

            node["reliability"] = (
                correct + 1
            ) / (
                total + 2
            )

            self.update_prediction_reliability(
                agent,
                y_true,
                pred
            )

        # Estimate each directed relationship.
        #
        # source -> target means:
        # How much does knowing the source prediction
        # improve prediction of the true label when the
        # target prediction is already known?

        for source in predictions:

            for target in predictions:

                if source == target:
                    continue

                baseline_loss = self._conditional_log_loss(
                    y_true,
                    predictions[target]
                )

                combined_loss = self._conditional_log_loss(
                    y_true,
                    predictions[target],
                    predictions[source]
                )

                information_gain = (
                    baseline_loss - combined_loss
                )

                # Negative gains do not create positive influence.
                information_gain = max(
                    0.0,
                    information_gain
                )

                edge = self.graph.edges[source, target]

                edge["information_gain"] = float(
                    information_gain
                )

                edge["base_weight"] = float(
                    information_gain
                )

        self.update_influence_weights()

        return self.get_graph_summary()

    # --------------------------------------------------
    # 6. UPDATE OVERALL RELIABILITY
    # --------------------------------------------------

    def update_reliability(
        self,
        agent_name,
        correct,
        total
    ):

        if agent_name not in self.graph:
            raise ValueError(
                f"Unknown agent: {agent_name}"
            )

        if (
            total < 0
            or correct < 0
            or correct > total
        ):
            raise ValueError(
                "Invalid correct/total values."
            )

        reliability = (
            correct + 1
        ) / (
            total + 2
        )

        self.graph.nodes[agent_name][
            "reliability"
        ] = reliability

        self.update_influence_weights()

    # --------------------------------------------------
    # 7. UPDATE CLASS-SPECIFIC RECALL
    # --------------------------------------------------

    def update_context_reliability(
        self,
        agent_name,
        normal_correct,
        normal_total,
        attack_correct,
        attack_total
    ):

        if agent_name not in self.graph:
            raise ValueError(
                f"Unknown agent: {agent_name}"
            )

        values = [
            normal_correct,
            normal_total,
            attack_correct,
            attack_total
        ]

        if any(value < 0 for value in values):
            raise ValueError(
                "Counts cannot be negative."
            )

        if normal_correct > normal_total:
            raise ValueError(
                "Normal correct cannot exceed total."
            )

        if attack_correct > attack_total:
            raise ValueError(
                "Attack correct cannot exceed total."
            )

        node = self.graph.nodes[agent_name]

        if normal_total > 0:

            node["normal_reliability"] = (
                normal_correct + 1
            ) / (
                normal_total + 2
            )

        if attack_total > 0:

            node["attack_reliability"] = (
                attack_correct + 1
            ) / (
                attack_total + 2
            )

        total_correct = (
            normal_correct + attack_correct
        )

        total_samples = (
            normal_total + attack_total
        )

        if total_samples > 0:

            node["reliability"] = (
                total_correct + 1
            ) / (
                total_samples + 2
            )

        self.update_influence_weights()

    # --------------------------------------------------
    # 8. UPDATE PREDICTION-SPECIFIC PRECISION
    # --------------------------------------------------

    def update_prediction_reliability(
        self,
        agent_name,
        y_true,
        predictions
    ):

        if agent_name not in self.graph:
            raise ValueError(
                f"Unknown agent: {agent_name}"
            )

        y_true = np.asarray(y_true, dtype=int)
        predictions = np.asarray(
            predictions, dtype=int
        )

        if len(y_true) != len(predictions):
            raise ValueError(
                "Labels and predictions must have equal lengths."
            )

        if len(y_true) == 0:
            raise ValueError(
                "Validation data cannot be empty."
            )

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                "Predictions must contain only 0 and 1."
            )

        node = self.graph.nodes[agent_name]

        for predicted_class in [0, 1]:

            mask = predictions == predicted_class

            predicted_count = int(mask.sum())

            correct_count = int(
                np.sum(
                    (y_true == predicted_class) & mask
                )
            )

            if predicted_count > 0:

                precision = (
                    correct_count + 1
                ) / (
                    predicted_count + 2
                )

                if predicted_class == 0:

                    node["normal_precision"] = precision

                else:

                    node["attack_precision"] = precision

        node["normal_samples"] = int(
            np.sum(predictions == 0)
        )

        node["attack_samples"] = int(
            np.sum(predictions == 1)
        )

        total_correct = int(
            np.sum(y_true == predictions)
        )

        node["reliability"] = (
            total_correct + 1
        ) / (
            len(y_true) + 2
        )

        self.update_influence_weights()

        return {
            "normal_precision": node["normal_precision"],
            "attack_precision": node["attack_precision"],
            "overall_reliability": node["reliability"],
            "normal_predictions": node["normal_samples"],
            "attack_predictions": node["attack_samples"]
        }

    # --------------------------------------------------
    # 9. GET RELIABILITY
    # --------------------------------------------------

    def get_context_reliability(
        self,
        agent_name,
        predicted_class
    ):

        if agent_name not in self.graph:
            raise ValueError(
                f"Unknown agent: {agent_name}"
            )

        if predicted_class == 0:
            return self.graph.nodes[agent_name][
                "normal_reliability"
            ]

        if predicted_class == 1:
            return self.graph.nodes[agent_name][
                "attack_reliability"
            ]

        raise ValueError(
            "Class must be 0 or 1."
        )

    def get_prediction_reliability(
        self,
        agent_name,
        predicted_class
    ):

        if agent_name not in self.graph:
            raise ValueError(
                f"Unknown agent: {agent_name}"
            )

        if predicted_class == 0:
            return self.graph.nodes[agent_name][
                "normal_precision"
            ]

        if predicted_class == 1:
            return self.graph.nodes[agent_name][
                "attack_precision"
            ]

        raise ValueError(
            "Class must be 0 or 1."
        )

    # --------------------------------------------------
    # 10. GET GRAPH SUMMARY
    # --------------------------------------------------

    def get_graph_summary(self):

        summary = {}

        for agent, data in self.graph.nodes(data=True):

            summary[agent] = {
                "reliability": data["reliability"],
                "influence_score": data["influence_score"]
            }

        return summary

    # --------------------------------------------------
    # 11. DISPLAY GRAPH
    # --------------------------------------------------

    def display_graph(self):

        print("\nDATA-DRIVEN DYNAMIC INFLUENCE GRAPH")
        print("=" * 65)

        for agent, data in self.graph.nodes(data=True):

            print(f"\n{agent}")

            print(
                f"Overall reliability: "
                f"{data['reliability']:.4f}"
            )

            print(
                f"Normal prediction precision: "
                f"{data['normal_precision']:.4f}"
            )

            print(
                f"Attack prediction precision: "
                f"{data['attack_precision']:.4f}"
            )

            print(
                f"Influence score: "
                f"{data['influence_score']:.4f}"
            )

        print("\nLEARNED DIRECTED RELATIONSHIPS")

        for source, target, data in self.graph.edges(data=True):

            print(
                f"{source} -> {target}: "
                f"Gain={data['information_gain']:.5f}, "
                f"Influence={data['influence']:.4f}"
            )
