
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer
from influence.influence_graph import DynamicInfluenceGraph


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
RESULT_PATH = "results/confidence_consensus_evaluation.csv"

CONFIDENCE_BINS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01)
]

SMOOTHING_STRENGTH = 200

os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATA AND CREATE THREE-WAY SPLIT
# --------------------------------------------------

print("\nLOADING DATA")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(
    columns=["label", "attack_cat", "id"]
)

y = df["label"].astype(int)

# First split: 60% training, 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: 20% calibration, 20% evaluation
X_calibration, X_evaluation, y_calibration, y_evaluation = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
)

y_cal = y_calibration.to_numpy()
y_eval = y_evaluation.to_numpy()

print("Training records:", len(X_train))
print("Calibration records:", len(X_calibration))
print("Evaluation records:", len(X_evaluation))


# --------------------------------------------------
# 2. TRAIN SPECIALIST AGENTS
# --------------------------------------------------

trainer = MLAgentTrainer()

models = {}

agent_names = list(AGENT_FEATURES.keys())

print("\nTRAINING SPECIALIST AGENTS")

for agent, features in AGENT_FEATURES.items():

    model = trainer.prepare_model(features, df)

    model.fit(
        X_train[features],
        y_train
    )

    models[agent] = model

    print(f"{agent}: Training completed")


# --------------------------------------------------
# 3. GENERATE CALIBRATION PREDICTIONS
# --------------------------------------------------

print("\nGENERATING CALIBRATION PREDICTIONS")

cal_predictions = {}
cal_probabilities = {}
cal_confidence = {}

for agent, features in AGENT_FEATURES.items():

    model = models[agent]

    raw_probabilities = model.predict_proba(
        X_calibration[features]
    )

    classes = list(model.classes_)

    probabilities = np.column_stack([
        raw_probabilities[:, classes.index(0)],
        raw_probabilities[:, classes.index(1)]
    ])

    cal_probabilities[agent] = probabilities

    cal_predictions[agent] = (
        probabilities[:, 1] > probabilities[:, 0]
    ).astype(int)

    cal_confidence[agent] = np.max(
        probabilities,
        axis=1
    )


# --------------------------------------------------
# 4. ESTIMATE HISTORICAL AND CONTEXTUAL RELIABILITY
# --------------------------------------------------

print("\nESTIMATING AGENT RELIABILITY")

reliability = {}
confidence_profiles = {}

for agent in agent_names:

    pred = cal_predictions[agent]

    confidence = cal_confidence[agent]

    correct = (pred == y_cal).astype(int)

    # Overall reliability with Laplace smoothing
    overall_reliability = (
        correct.sum() + 1
    ) / (
        len(y_cal) + 2
    )

    # Prediction-specific precision
    class_precision = {}

    for predicted_class in [0, 1]:

        mask = pred == predicted_class

        count = int(mask.sum())

        correct_count = int(
            np.sum((pred == predicted_class) & (y_cal == predicted_class))
        )

        class_precision[predicted_class] = (
            (correct_count + 1) / (count + 2)
        )

    reliability[agent] = {
        "overall": float(overall_reliability),
        "normal_precision": float(class_precision[0]),
        "attack_precision": float(class_precision[1])
    }

    # Confidence-bin conditional accuracy
    profile = {}

    for lower, upper in CONFIDENCE_BINS:

        mask = (
            (confidence >= lower) &
            (confidence < upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        bin_accuracy = float(
            correct[mask].mean()
        )

        # Shrink bin accuracy towards overall accuracy
        smoothed_accuracy = (
            count * bin_accuracy
            + SMOOTHING_STRENGTH * overall_reliability
        ) / (
            count + SMOOTHING_STRENGTH
        )

        profile[(lower, upper)] = float(
            smoothed_accuracy
        )

    confidence_profiles[agent] = profile

    print(f"\n{agent}")
    print(f"Overall reliability: {overall_reliability:.4f}")
    print(f"Normal precision: {class_precision[0]:.4f}")
    print(f"Attack precision: {class_precision[1]:.4f}")


# --------------------------------------------------
# 5. INITIALIZE INFLUENCE GRAPH
# --------------------------------------------------

# --------------------------------------------------
# 5. LEARN DATA-DRIVEN INFLUENCE GRAPH
# --------------------------------------------------

print("\nLEARNING DATA-DRIVEN INFLUENCE GRAPH")

graph = DynamicInfluenceGraph()

# Learn directed relationships using calibration data only
influence_summary = graph.estimate_influence_from_predictions(
    y_cal,
    cal_predictions
)

print("\nLEARNED INFLUENCE SUMMARY")

for agent, values in influence_summary.items():

    print(f"\n{agent}")
    print("Reliability:", values["reliability"])
    print("Influence score:", values["influence_score"])

graph.display_graph()


# Calculate normalized outgoing graph influence
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

if max_outgoing > 0:

    graph_factors = {
        agent: outgoing[agent] / max_outgoing
        for agent in agent_names
    }

else:

    graph_factors = {
        agent: 1.0
        for agent in agent_names
    }

print("\nLEARNED GRAPH FACTORS")

for agent, factor in graph_factors.items():
    print(f"{agent}: {factor:.4f}")


# --------------------------------------------------
# 6. CONFIDENCE LOOKUP FUNCTION
# --------------------------------------------------

def get_confidence_reliability(agent, confidence):

    profile = confidence_profiles[agent]

    for (lower, upper), value in profile.items():

        if lower <= confidence < upper:
            return value

    # Fallback to overall calibration reliability
    return reliability[agent]["overall"]


# --------------------------------------------------
# 7. GENERATE INDEPENDENT EVALUATION PREDICTIONS
# --------------------------------------------------

print("\nGENERATING EVALUATION PREDICTIONS")

eval_predictions = {}
eval_probabilities = {}
eval_confidence = {}

for agent, features in AGENT_FEATURES.items():

    model = models[agent]

    raw_probabilities = model.predict_proba(
        X_evaluation[features]
    )

    classes = list(model.classes_)

    probabilities = np.column_stack([
        raw_probabilities[:, classes.index(0)],
        raw_probabilities[:, classes.index(1)]
    ])

    eval_probabilities[agent] = probabilities

    eval_predictions[agent] = (
        probabilities[:, 1] > probabilities[:, 0]
    ).astype(int)

    eval_confidence[agent] = np.max(
        probabilities,
        axis=1
    )

    print(f"{agent}: Completed")


# --------------------------------------------------
# 8. MAJORITY VOTING BASELINE
# --------------------------------------------------

prediction_matrix = np.column_stack([
    eval_predictions[agent]
    for agent in agent_names
])

majority_predictions = (
    prediction_matrix.sum(axis=1) > len(agent_names) / 2
).astype(int)


# --------------------------------------------------
# 9. EXPERIMENT A-E: CALCULATE WEIGHTS
# --------------------------------------------------

print("\nCALCULATING FIVE WEIGHTING STRATEGIES")

n_samples = len(X_evaluation)

weight_matrices = {
    "A - Historical Reliability": np.zeros(
        (n_samples, len(agent_names))
    ),
    "B - Contextual Reliability": np.zeros(
        (n_samples, len(agent_names))
    ),
    "C - Confidence Reliability": np.zeros(
        (n_samples, len(agent_names))
    ),
    "D - Context + Confidence": np.zeros(
        (n_samples, len(agent_names))
    ),
    "E - Full Influence-Aware": np.zeros(
        (n_samples, len(agent_names))
    )
}

for index, agent in enumerate(agent_names):

    predicted = eval_predictions[agent]

    confidence = eval_confidence[agent]

    historical = reliability[agent]["overall"]

    contextual = np.where(
        predicted == 0,
        reliability[agent]["normal_precision"],
        reliability[agent]["attack_precision"]
    )

    confidence_factor = np.array([
        get_confidence_reliability(agent, value)
        for value in confidence
    ])

    graph_factor = graph_factors[agent]

    weight_matrices[
        "A - Historical Reliability"
    ][:, index] = historical

    weight_matrices[
        "B - Contextual Reliability"
    ][:, index] = contextual

    weight_matrices[
        "C - Confidence Reliability"
    ][:, index] = confidence_factor

    weight_matrices[
        "D - Context + Confidence"
    ][:, index] = (
        contextual * confidence_factor
    )

    weight_matrices[
        "E - Full Influence-Aware"
    ][:, index] = (
        contextual
        * confidence_factor
        * (0.5 + 0.5 * graph_factor)
    )


# --------------------------------------------------
# 10. NORMALIZE WEIGHTS AND AGGREGATE PROBABILITIES
# --------------------------------------------------

methods = {
    "Majority Voting MAS": majority_predictions
}

print("\nGENERATING CONSENSUS DECISIONS")

for method_name, weight_matrix in weight_matrices.items():

    row_sums = weight_matrix.sum(axis=1, keepdims=True)

    normalized_weights = (
        weight_matrix /
        np.maximum(row_sums, 1e-12)
    )

    normal_scores = np.zeros(n_samples)
    attack_scores = np.zeros(n_samples)

    for index, agent in enumerate(agent_names):

        normal_scores += (
            normalized_weights[:, index]
            * eval_probabilities[agent][:, 0]
        )

        attack_scores += (
            normalized_weights[:, index]
            * eval_probabilities[agent][:, 1]
        )

    methods[method_name] = (
        attack_scores > normal_scores
    ).astype(int)


# --------------------------------------------------
# 11. EVALUATE ALL METHODS
# --------------------------------------------------

print("\nINDEPENDENT VALIDATION EVALUATION")
print("=" * 85)

results = []

for method_name, y_pred in methods.items():

    accuracy = accuracy_score(y_eval, y_pred)

    macro_f1 = f1_score(
        y_eval,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision = precision_score(
        y_eval,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_eval,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    fnr = fn / (fn + tp) if (fn + tp) else 0

    results.append({
        "Method": method_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "FNR": fnr
    })

    print(f"\n{method_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")


# --------------------------------------------------
# 12. COMPARE AGAINST MAJORITY VOTING
# --------------------------------------------------

print("\nDECISION COMPARISON")
print("=" * 60)

for method_name, y_pred in methods.items():

    different = int(
        np.sum(y_pred != majority_predictions)
    )

    print(f"\n{method_name}")
    print("Different decisions:", different)
    print("Identical decisions:", n_samples - different)


# --------------------------------------------------
# 13. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nRESULTS SAVED TO:")
print(RESULT_PATH)

print("\nCONTROLLED CONSENSUS EXPERIMENT COMPLETED")

# --------------------------------------------------
# 13. STATISTICAL SIGNIFICANCE ANALYSIS
# --------------------------------------------------

print("\nSTATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 85)

BOOTSTRAP_ITERATIONS = 2000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

baseline = majority_predictions

statistical_results = []
raw_p_values = []

weighted_methods = [
    name for name in methods
    if name != "Majority Voting MAS"
]

def calculate_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )


for method_name in weighted_methods:

    method_predictions = methods[method_name]

    # ----------------------------------------------
    # A. EXACT McNEMAR TEST
    # ----------------------------------------------

    baseline_correct = baseline == y_eval
    method_correct = method_predictions == y_eval

    # Baseline correct, proposed method incorrect
    b = int(np.sum(baseline_correct & ~method_correct))

    # Baseline incorrect, proposed method correct
    c = int(np.sum(~baseline_correct & method_correct))

    discordant_pairs = b + c

    if discordant_pairs > 0:
        mcnemar_p = binomtest(
            min(b, c),
            discordant_pairs,
            p=0.5,
            alternative="two-sided"
        ).pvalue
    else:
        mcnemar_p = 1.0

    raw_p_values.append(mcnemar_p)

    # ----------------------------------------------
    # B. PAIRED BOOTSTRAP CONFIDENCE INTERVALS
    # ----------------------------------------------

    accuracy_differences = []
    f1_differences = []

    baseline_accuracy, baseline_f1 = calculate_metrics(
        y_eval,
        baseline
    )

    for _ in range(BOOTSTRAP_ITERATIONS):

        indices = rng.integers(
            0,
            len(y_eval),
            size=len(y_eval)
        )

        sampled_y = y_eval[indices]

        sampled_baseline = baseline[indices]

        sampled_method = method_predictions[indices]

        base_acc, base_f1 = calculate_metrics(
            sampled_y,
            sampled_baseline
        )

        method_acc, method_f1 = calculate_metrics(
            sampled_y,
            sampled_method
        )

        accuracy_differences.append(
            method_acc - base_acc
        )

        f1_differences.append(
            method_f1 - base_f1
        )

    accuracy_differences = np.array(
        accuracy_differences
    )

    f1_differences = np.array(
        f1_differences
    )

    accuracy_ci = np.percentile(
        accuracy_differences,
        [2.5, 97.5]
    )

    f1_ci = np.percentile(
        f1_differences,
        [2.5, 97.5]
    )

    accuracy_improvement = (
        accuracy_score(y_eval, method_predictions)
        - baseline_accuracy
    )

    f1_improvement = (
        f1_score(
            y_eval,
            method_predictions,
            average="macro",
            zero_division=0
        )
        - baseline_f1
    )

    statistical_results.append({
        "Method": method_name,
        "Baseline_Correct_Method_Wrong": b,
        "Baseline_Wrong_Method_Correct": c,
        "Discordant_Pairs": discordant_pairs,
        "McNemar_p_value": mcnemar_p,
        "Accuracy_Improvement": accuracy_improvement,
        "Accuracy_CI_Lower": accuracy_ci[0],
        "Accuracy_CI_Upper": accuracy_ci[1],
        "Macro_F1_Improvement": f1_improvement,
        "Macro_F1_CI_Lower": f1_ci[0],
        "Macro_F1_CI_Upper": f1_ci[1]
    })


# ----------------------------------------------
# C. HOLM MULTIPLE-TEST CORRECTION
# ----------------------------------------------

number_of_tests = len(raw_p_values)

sorted_indices = np.argsort(raw_p_values)

adjusted_p_values = np.zeros(number_of_tests)

running_max = 0.0

for rank, original_index in enumerate(sorted_indices):

    adjusted = (
        number_of_tests - rank
    ) * raw_p_values[original_index]

    running_max = max(running_max, adjusted)

    adjusted_p_values[original_index] = min(
        running_max,
        1.0
    )


# ----------------------------------------------
# D. DISPLAY RESULTS
# ----------------------------------------------

print("\nPAIRED STATISTICAL RESULTS")
print("=" * 85)

for index, result in enumerate(statistical_results):

    result["Holm_Adjusted_p_value"] = (
        adjusted_p_values[index]
    )

    result["McNemar_Significant"] = (
        adjusted_p_values[index] < 0.05
    )

    accuracy_ci_excludes_zero = (
        result["Accuracy_CI_Lower"] > 0
        or result["Accuracy_CI_Upper"] < 0
    )

    f1_ci_excludes_zero = (
        result["Macro_F1_CI_Lower"] > 0
        or result["Macro_F1_CI_Upper"] < 0
    )

    print(f"\n{result['Method']}")

    print(
        "McNemar p-value:",
        f"{result['McNemar_p_value']:.6f}"
    )

    print(
        "Holm-adjusted p-value:",
        f"{result['Holm_Adjusted_p_value']:.6f}"
    )

    print(
        "Accuracy improvement:",
        f"{result['Accuracy_Improvement'] * 100:.4f} percentage points"
    )

    print(
        "Accuracy 95% CI:",
        f"[{result['Accuracy_CI_Lower'] * 100:.4f}%, "
        f"{result['Accuracy_CI_Upper'] * 100:.4f}%]"
    )

    print(
        "Macro F1 improvement:",
        f"{result['Macro_F1_Improvement'] * 100:.4f} percentage points"
    )

    print(
        "Macro F1 95% CI:",
        f"[{result['Macro_F1_CI_Lower'] * 100:.4f}%, "
        f"{result['Macro_F1_CI_Upper'] * 100:.4f}%]"
    )

    print(
        "McNemar significant after Holm correction:",
        result["McNemar_Significant"]
    )

    print(
        "Accuracy CI excludes zero:",
        accuracy_ci_excludes_zero
    )

    print(
        "Macro F1 CI excludes zero:",
        f1_ci_excludes_zero
    )


# --------------------------------------------------
# 14. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

statistical_df = pd.DataFrame(statistical_results)

results_df = results_df.merge(
    statistical_df,
    on="Method",
    how="left"
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nRESULTS SAVED TO:")
print(RESULT_PATH)

print("\nSTATISTICAL ANALYSIS COMPLETED")

# --------------------------------------------------
# 14. DIRECT PAIRED ABLATION ANALYSIS
# --------------------------------------------------

print("\nDIRECT PAIRED ABLATION ANALYSIS")
print("=" * 85)

ablation_pairs = [
    (
        "A vs B - Contextual Reliability",
        "A - Historical Reliability",
        "B - Contextual Reliability"
    ),
    (
        "A vs C - Confidence Reliability",
        "A - Historical Reliability",
        "C - Confidence Reliability"
    ),
    (
        "C vs D - Additional Context",
        "C - Confidence Reliability",
        "D - Context + Confidence"
    ),
    (
        "D vs E - Graph Influence",
        "D - Context + Confidence",
        "E - Full Influence-Aware"
    )
]

ablation_results = []
ablation_p_values = []

for comparison, baseline_name, proposed_name in ablation_pairs:

    baseline_pred = methods[baseline_name]
    proposed_pred = methods[proposed_name]

    # McNemar test on paired correctness
    baseline_correct = baseline_pred == y_eval
    proposed_correct = proposed_pred == y_eval

    b = int(np.sum(baseline_correct & ~proposed_correct))
    c = int(np.sum(~baseline_correct & proposed_correct))

    discordant = b + c

    if discordant > 0:
        p_value = binomtest(
            min(b, c),
            discordant,
            p=0.5,
            alternative="two-sided"
        ).pvalue
    else:
        p_value = 1.0

    ablation_p_values.append(p_value)

    # Paired bootstrap
    accuracy_diffs = []
    f1_diffs = []

    for _ in range(BOOTSTRAP_ITERATIONS):

        indices = rng.integers(
            0,
            len(y_eval),
            size=len(y_eval)
        )

        sampled_y = y_eval[indices]

        base_acc, base_f1 = calculate_metrics(
            sampled_y,
            baseline_pred[indices]
        )

        prop_acc, prop_f1 = calculate_metrics(
            sampled_y,
            proposed_pred[indices]
        )

        accuracy_diffs.append(prop_acc - base_acc)
        f1_diffs.append(prop_f1 - base_f1)

    accuracy_diffs = np.asarray(accuracy_diffs)
    f1_diffs = np.asarray(f1_diffs)

    accuracy_ci = np.percentile(
        accuracy_diffs, [2.5, 97.5]
    )

    f1_ci = np.percentile(
        f1_diffs, [2.5, 97.5]
    )

    accuracy_difference = (
        accuracy_score(y_eval, proposed_pred)
        - accuracy_score(y_eval, baseline_pred)
    )

    f1_difference = (
        f1_score(
            y_eval, proposed_pred,
            average="macro", zero_division=0
        )
        - f1_score(
            y_eval, baseline_pred,
            average="macro", zero_division=0
        )
    )

    ablation_results.append({
        "Comparison": comparison,
        "Baseline": baseline_name,
        "Proposed": proposed_name,
        "Baseline_Correct_Proposed_Wrong": b,
        "Baseline_Wrong_Proposed_Correct": c,
        "Discordant_Pairs": discordant,
        "McNemar_p_value": p_value,
        "Accuracy_Difference": accuracy_difference,
        "Accuracy_CI_Lower": accuracy_ci[0],
        "Accuracy_CI_Upper": accuracy_ci[1],
        "Macro_F1_Difference": f1_difference,
        "Macro_F1_CI_Lower": f1_ci[0],
        "Macro_F1_CI_Upper": f1_ci[1]
    })


# Holm correction for four comparisons
sorted_indices = np.argsort(ablation_p_values)

adjusted_values = np.zeros(len(ablation_p_values))
running_max = 0.0

for rank, original_index in enumerate(sorted_indices):

    adjusted = (
        len(ablation_p_values) - rank
    ) * ablation_p_values[original_index]

    running_max = max(running_max, adjusted)

    adjusted_values[original_index] = min(
        running_max, 1.0
    )


# Display results
print("\nABLATION RESULTS")
print("=" * 85)

for index, result in enumerate(ablation_results):

    adjusted_p = adjusted_values[index]

    result["Holm_Adjusted_p_value"] = adjusted_p

    accuracy_ci_excludes_zero = (
        result["Accuracy_CI_Lower"] > 0
        or result["Accuracy_CI_Upper"] < 0
    )

    f1_ci_excludes_zero = (
        result["Macro_F1_CI_Lower"] > 0
        or result["Macro_F1_CI_Upper"] < 0
    )

    print(f"\n{result['Comparison']}")

    print(
        "Accuracy difference:",
        f"{result['Accuracy_Difference'] * 100:.4f} pp"
    )

    print(
        "Accuracy 95% CI:",
        f"[{result['Accuracy_CI_Lower'] * 100:.4f}%, "
        f"{result['Accuracy_CI_Upper'] * 100:.4f}%]"
    )

    print(
        "Macro F1 difference:",
        f"{result['Macro_F1_Difference'] * 100:.4f} pp"
    )

    print(
        "Macro F1 95% CI:",
        f"[{result['Macro_F1_CI_Lower'] * 100:.4f}%, "
        f"{result['Macro_F1_CI_Upper'] * 100:.4f}%]"
    )

    print(f"McNemar p-value: {result['McNemar_p_value']:.8g}")
    print(f"Holm-adjusted p-value: {adjusted_p:.8g}")

    print(
        "Accuracy CI excludes zero:",
        accuracy_ci_excludes_zero
    )

    print(
        "Macro F1 CI excludes zero:",
        f1_ci_excludes_zero
    )


# Save ablation results separately
ablation_df = pd.DataFrame(ablation_results)

ablation_df.to_csv(
    "results/ablation_pairwise_statistics.csv",
    index=False
)

print("\nABLATION RESULTS SAVED TO:")
print("results/ablation_pairwise_statistics.csv")

print("\nDIRECT ABLATION ANALYSIS COMPLETED")


