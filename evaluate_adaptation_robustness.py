
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wilcoxon

from agents.ml_agents import MLAgentTrainer
from agents.feature_groups import AGENT_FEATURES
from influence.influence_graph import DynamicInfluenceGraph


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

N_TRIALS = 10
BATCH_SIZE = 500
ALPHA = 0.30
DEGRADATION = 0.30
BASE_SEED = 42

agent_names = list(AGENT_FEATURES.keys())


# --------------------------------------------------
# 1. LOAD AND SPLIT DATA
# --------------------------------------------------

print("\nLOADING DATA")

df = pd.read_csv(
    "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
)

X = df.drop(columns=["label", "attack_cat", "id"])
y = df["label"].astype(int).to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    stratify=y,
    random_state=42
)

X_cal, X_eval, y_cal, y_eval = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Training:", len(y_train))
print("Calibration:", len(y_cal))
print("Evaluation:", len(y_eval))


# --------------------------------------------------
# 2. TRAIN AGENTS ONCE
# --------------------------------------------------

print("\nTRAINING SPECIALIST AGENTS")

trainer = MLAgentTrainer()
models = {}

for agent, features in AGENT_FEATURES.items():

    model = trainer.prepare_model(features, df)
    model.fit(X_train[features], y_train)

    models[agent] = model

    print(f"{agent}: Training completed")


# --------------------------------------------------
# 3. CALIBRATION PREDICTIONS
# --------------------------------------------------

print("\nCALIBRATING AGENTS")

cal_predictions = {}

for agent, features in AGENT_FEATURES.items():

    cal_predictions[agent] = models[agent].predict(
        X_cal[features]
    )


# --------------------------------------------------
# 4. LEARN INFLUENCE GRAPH
# --------------------------------------------------

print("\nLEARNING INFLUENCE GRAPH")

graph = DynamicInfluenceGraph()

calibration_summary = graph.estimate_influence_from_predictions(
    y_cal,
    cal_predictions
)

base_reliability = {
    agent: calibration_summary[agent]["reliability"]
    for agent in agent_names
}

outgoing = {}

for agent in agent_names:

    outgoing[agent] = sum(
        data["influence"]
        for _, _, data in graph.graph.out_edges(
            agent,
            data=True
        )
    )

max_outgoing = max(outgoing.values(), default=0)

graph_factors = {
    agent: (
        outgoing[agent] / max_outgoing
        if max_outgoing > 0 else 1.0
    )
    for agent in agent_names
}


# --------------------------------------------------
# 5. EVALUATION PREDICTIONS
# --------------------------------------------------

print("\nGENERATING EVALUATION PREDICTIONS")

eval_predictions = {}

for agent, features in AGENT_FEATURES.items():

    eval_predictions[agent] = models[agent].predict(
        X_eval[features]
    )

    print(f"{agent}: Completed")


# --------------------------------------------------
# 6. CONSENSUS FUNCTIONS
# --------------------------------------------------

def majority_vote(predictions):

    matrix = np.column_stack(
        [predictions[a] for a in agent_names]
    )

    return (
        matrix.sum(axis=1) > len(agent_names) / 2
    ).astype(int)


def weighted_vote(predictions, weights):

    total_weight = sum(weights.values())

    if total_weight <= 0:
        return majority_vote(predictions)

    scores = np.zeros(
        len(next(iter(predictions.values())))
    )

    for agent in agent_names:
        scores += weights[agent] * predictions[agent]

    return (
        scores > total_weight / 2
    ).astype(int)


def calculate_metrics(y_true, predictions):

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# --------------------------------------------------
# 7. RUN MULTIPLE CORRUPTION TRIALS
# --------------------------------------------------

print("\nRUNNING ROBUSTNESS EXPERIMENT")

all_results = []

midpoint = len(y_eval) // 2

fixed_weights = {
    agent: base_reliability[agent] *
    (0.5 + 0.5 * graph_factors[agent])
    for agent in agent_names
}

for target_agent in agent_names:

    print("\n" + "=" * 60)
    print("DEGRADED AGENT:", target_agent)
    print("=" * 60)

    for trial in range(N_TRIALS):

        seed = BASE_SEED + trial
        rng = np.random.default_rng(seed)

        adaptive_reliability = base_reliability.copy()

        # Select a new corruption pattern for this trial
        corrupted_indices = set(
            rng.choice(
                midpoint,
                size=int(midpoint * DEGRADATION),
                replace=False
            ).tolist()
        )

        predictions_by_method = {
            "Majority Voting": [],
            "Fixed Influence": [],
            "Adaptive Influence": []
        }

        # Process evaluation data in sequential batches
        for start in range(0, len(y_eval), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(y_eval))

            batch_labels = y_eval[start:end]

            batch_predictions = {
                agent: eval_predictions[agent][start:end].copy()
                for agent in agent_names
            }

            # Corrupt 30% of the target agent's predictions
            # in the first half of the evaluation sequence.

            if start < midpoint:

                for local_index, global_index in enumerate(
                    range(start, end)
                ):

                    if global_index in corrupted_indices:

                        batch_predictions[target_agent][
                            local_index
                        ] = 1 - batch_predictions[target_agent][
                            local_index
                        ]

            # Make decisions using weights available BEFORE
            # feedback from this batch is revealed.

            majority_batch = majority_vote(batch_predictions)

            fixed_batch = weighted_vote(
                batch_predictions,
                fixed_weights
            )

            adaptive_weights = {
                agent: adaptive_reliability[agent] *
                (0.5 + 0.5 * graph_factors[agent])
                for agent in agent_names
            }

            adaptive_batch = weighted_vote(
                batch_predictions,
                adaptive_weights
            )

            predictions_by_method["Majority Voting"].extend(
                majority_batch.tolist()
            )

            predictions_by_method["Fixed Influence"].extend(
                fixed_batch.tolist()
            )

            predictions_by_method["Adaptive Influence"].extend(
                adaptive_batch.tolist()
            )

            # Delayed feedback:
            # update reliability only AFTER current decisions.

            for agent in agent_names:

                observed_accuracy = accuracy_score(
                    batch_labels,
                    batch_predictions[agent]
                )

                adaptive_reliability[agent] = (
                    (1 - ALPHA) * adaptive_reliability[agent]
                    + ALPHA * observed_accuracy
                )

        # Evaluate complete trial

        for method, predictions in predictions_by_method.items():

            metrics = calculate_metrics(
                y_eval,
                np.array(predictions)
            )

            all_results.append({
                "target_agent": target_agent,
                "trial": trial + 1,
                "seed": seed,
                "method": method,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"]
            })

        print(
            f"Trial {trial + 1}/{N_TRIALS} completed"
        )


# --------------------------------------------------
# 8. SAVE RAW RESULTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

raw_df = pd.DataFrame(all_results)

raw_path = "results/adaptation_robustness_trials.csv"

raw_df.to_csv(raw_path, index=False)

print("\nRAW RESULTS SAVED:", raw_path)


# --------------------------------------------------
# 9. MEAN AND STANDARD DEVIATION
# --------------------------------------------------

summary_df = (
    raw_df
    .groupby(["target_agent", "method"])
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std")
    )
    .reset_index()
)

summary_path = "results/adaptation_robustness_summary.csv"

summary_df.to_csv(summary_path, index=False)

print("\nMEAN PERFORMANCE ACROSS TRIALS")
print(summary_df.to_string(index=False))

print("\nSUMMARY SAVED:", summary_path)


# --------------------------------------------------
# 10. PAIRED STATISTICAL COMPARISONS
# --------------------------------------------------

print("\nPAIRED WILCOXON TESTS")

statistical_results = []

for target_agent in agent_names:

    scenario = raw_df[
        raw_df["target_agent"] == target_agent
    ]

    for metric in ["accuracy", "macro_f1"]:

        pivot = scenario.pivot(
            index="trial",
            columns="method",
            values=metric
        )

        for comparison in [
            "Majority Voting",
            "Fixed Influence"
        ]:

            adaptive = pivot["Adaptive Influence"]
            baseline = pivot[comparison]

            differences = adaptive - baseline

            if np.allclose(differences, 0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    adaptive,
                    baseline,
                    alternative="two-sided"
                )

                statistic = test.statistic
                p_value = test.pvalue

            statistical_results.append({
                "target_agent": target_agent,
                "metric": metric,
                "comparison": f"Adaptive vs {comparison}",
                "mean_difference": differences.mean(),
                "wilcoxon_statistic": statistic,
                "p_value": p_value
            })

            print(
                f"{target_agent} | {metric} | "
                f"Adaptive vs {comparison}: "
                f"Mean difference={differences.mean():.6f}, "
                f"p={p_value:.4f}"
            )


stats_df = pd.DataFrame(statistical_results)

stats_path = "results/adaptation_robustness_statistics.csv"

stats_df.to_csv(stats_path, index=False)

print("\nSTATISTICS SAVED:", stats_path)

print("\nROBUSTNESS EXPERIMENT COMPLETED")
