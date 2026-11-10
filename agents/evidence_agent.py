
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd


class EvidenceAgent:

    def __init__(self, file_path):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        with open(file_path, "r", encoding="utf-8") as file:
            self.documents = [
                line.strip()
                for line in file
                if line.strip()
            ]

        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(
            self.embeddings.astype("float32")
        )

        # Statistical evidence reference
        self.feature_profiles = {}
        self.is_fitted = False

        self.feature_groups = {
            "Traffic Agent": [
                "rate", "sload", "dload",
                "dpkts", "spkts", "sbytes"
            ],
            "Threat Agent": [
                "sttl", "dttl"
            ],
            "Behaviour Agent": [
                "ct_dst_sport_ltm",
                "ct_src_dport_ltm"
            ],
            "Risk Agent": [
                "ct_state_ttl",
                "ct_dst_src_ltm"
            ]
        }

    # ---------------------------------------
    # EXISTING SEMANTIC EVIDENCE RETRIEVAL
    # ---------------------------------------

    def retrieve(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            min(top_k, len(self.documents))
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "evidence": self.documents[idx],
                "similarity": float(score)
            })

        return results

    # ---------------------------------------
    # TRAIN STATISTICAL EVIDENCE REFERENCE
    # ---------------------------------------

    def fit_reference(self, training_df, target_column="label"):

        if target_column not in training_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        if not set(training_df[target_column].dropna().unique()).issubset({0, 1}):
            raise ValueError("Target must contain binary labels 0 and 1.")

        self.feature_profiles = {}

        all_features = sorted({
            feature
            for features in self.feature_groups.values()
            for feature in features
        })

        for feature in all_features:

            if feature not in training_df.columns:
                continue

            values = pd.to_numeric(
                training_df[feature],
                errors="coerce"
            )

            valid = values.notna() & training_df[target_column].notna()

            values = values[valid].to_numpy(dtype=float)
            labels = training_df.loc[valid, target_column].to_numpy()

            normal_values = values[labels == 0]
            attack_values = values[labels == 1]

            if len(normal_values) == 0 or len(attack_values) == 0:
                continue

            # Learn quantile boundaries using training data only.
            edges = np.unique(
                np.quantile(
                    values,
                    np.linspace(0, 1, 11)
                )
            )

            # A constant feature cannot provide useful evidence.
            if len(edges) < 3:
                continue

            normal_bins = np.histogram(
                normal_values,
                bins=edges
            )[0].astype(float)

            attack_bins = np.histogram(
                attack_values,
                bins=edges
            )[0].astype(float)

            # Laplace smoothing prevents zero probabilities.
            normal_prob = (
                normal_bins + 1.0
            ) / (
                normal_bins.sum() + len(normal_bins)
            )

            attack_prob = (
                attack_bins + 1.0
            ) / (
                attack_bins.sum() + len(attack_bins)
            )

            self.feature_profiles[feature] = {
                "edges": edges,
                "normal_prob": normal_prob,
                "attack_prob": attack_prob
            }

        self.is_fitted = True

        print("\nSTATISTICAL EVIDENCE REFERENCE")
        print("=" * 45)
        print("Features learned:", len(self.feature_profiles))

        for agent_name, features in self.feature_groups.items():

            available = [
                feature
                for feature in features
                if feature in self.feature_profiles
            ]

            print(
                f"{agent_name}: {len(available)} features"
            )

    # ---------------------------------------
    # RECORD-SPECIFIC EVIDENCE GENERATION
    # ---------------------------------------

   
    def generate_record_evidence(self, record):

        if not self.is_fitted:
            raise RuntimeError(
                "Call fit_reference() before generating evidence."
            )

        evidence = []

        for agent_name, features in self.feature_groups.items():

            for feature in features:

                if feature not in self.feature_profiles:
                    continue

                if feature not in record.index:
                    continue

                try:
                    value = float(record[feature])
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(value):
                    continue

                profile = self.feature_profiles[feature]
                edges = profile["edges"]

                bin_index = np.searchsorted(
                    edges[1:-1],
                    value,
                    side="right"
                )

                bin_index = min(
                    bin_index,
                    len(profile["normal_prob"]) - 1
                )

                normal_probability = profile[
                    "normal_prob"
                ][bin_index]

                attack_probability = profile[
                    "attack_prob"
                ][bin_index]

                log_likelihood_ratio = np.log(
                    attack_probability / normal_probability
                )

                if abs(log_likelihood_ratio) < 0.40:
                    continue

                direction = (
                    "attack"
                    if log_likelihood_ratio > 0
                    else "normal"
                )

                evidence.append({
                    "evidence": (
                        f"{feature} has value {value:.4f}; "
                        f"its training-derived distribution "
                        f"supports {direction} traffic."
                    ),
                    "feature": feature,
                    "agent": agent_name,
                    "value": value,
                    "direction": direction,
                    "evidence_strength": min(
                        abs(float(log_likelihood_ratio)),
                        2.5
                    ),
                    "record_specific": True,
                    "feature_match": True
                })

        return evidence

    # ---------------------------------------
    # AGENT-LEVEL EVIDENCE AGGREGATION
    # ---------------------------------------

    def aggregate_agent_evidence(self, evidence):

        agent_results = {}

        for agent_name in self.feature_groups:

            agent_items = [
                item for item in evidence
                if item.get("agent") == agent_name
            ]

            attack_strength = sum(
                item["evidence_strength"]
                for item in agent_items
                if item["direction"] == "attack"
            )

            normal_strength = sum(
                item["evidence_strength"]
                for item in agent_items
                if item["direction"] == "normal"
            )

            attack_count = sum(
                item["direction"] == "attack"
                for item in agent_items
            )

            normal_count = sum(
                item["direction"] == "normal"
                for item in agent_items
            )

            total_count = attack_count + normal_count

            decision = None

            # Require at least two feature observations.
            if total_count >= 2:

                margin = abs(
                    attack_strength - normal_strength
                )

                if margin >= 0.75:

                    decision = (
                        1 if attack_strength > normal_strength
                        else 0
                    )

            agent_results[agent_name] = {
                "decision": decision,
                "attack_strength": round(attack_strength, 4),
                "normal_strength": round(normal_strength, 4),
                "attack_features": attack_count,
                "normal_features": normal_count,
                "total_features": total_count
            }

        return agent_results
