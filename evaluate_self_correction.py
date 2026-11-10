
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

from agents.feature_groups import AGENT_FEATURES
from agents.ml_agents import MLAgentTrainer
from agents.evidence_agent import EvidenceAgent
from agents.critic_agent import CriticAgent
from correction.self_correction import SelfCorrection


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
EVIDENCE_PATH = "datasets/evidence/sample_evidence.txt"

RESULT_PATH = "results/self_correction_evaluation.csv"
SUMMARY_PATH = "results/self_correction_summary.csv"

SAMPLE_SIZE = 500
RANDOM_STATE = 42


def majority_vote(opinions):

    votes = list(opinions.values())

    return int(sum(votes) > len(votes) / 2)


def calculate_metrics(y_true, predictions):

    return {
        "Accuracy": accuracy_score(y_true, predictions),

        "Macro F1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    }


def main():

    os.makedirs("results", exist_ok=True)

    print("\nLOADING UNSW-NB15 DATA")
    print("=" * 50)

    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(
        columns=["label", "attack_cat", "id"]
    )

    y = df["label"].astype(int)

    # 60% training, 20% calibration, 20% evaluation.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Reproducible evaluation subset.
    X_eval = X_eval.sample(
        n=min(SAMPLE_SIZE, len(X_eval)),
        random_state=RANDOM_STATE
    )

    y_eval = y_eval.loc[X_eval.index]

    print("Training records:", len(X_train))
    print("Calibration records:", len(X_cal))
    print("Evaluation sample:", len(X_eval))

    # -----------------------------------------
    # TRAIN SPECIALIST ML AGENTS
    # -----------------------------------------

    print("\nTRAINING SPECIALIST AGENTS")
    print("=" * 50)

    trainer = MLAgentTrainer()

    models = {}

    for agent, features in AGENT_FEATURES.items():

        model = trainer.prepare_model(features, df)

        model.fit(
            X_train[features],
            y_train
        )

        models[agent] = model

        print(agent, "trained successfully")

    # -----------------------------------------
    # GENERATE PREDICTIONS AND PROBABILITIES
    # -----------------------------------------

    print("\nGENERATING AGENT PREDICTIONS AND CONFIDENCE")
    print("=" * 50)

    predictions = {}
    probabilities = {}

    for agent, features in AGENT_FEATURES.items():

        model = models[agent]

        X_agent = X_eval[features]

        predictions[agent] = model.predict(
            X_agent
        ).astype(int)

        # Probability of the ATTACK class (label 1).
        class_index = list(model.classes_).index(1)

        probabilities[agent] = model.predict_proba(
            X_agent
        )[:, class_index]

        print(agent, "predictions and probabilities generated")

    agent_names = list(AGENT_FEATURES.keys())
        # -----------------------------------------
    # GENERATE CALIBRATION PREDICTIONS
    # -----------------------------------------

    print("\nGENERATING CALIBRATION PREDICTIONS")
    print("=" * 50)

    calibration_predictions = {}
    calibration_probabilities = {}

    for agent, features in AGENT_FEATURES.items():

        model = models[agent]

        calibration_predictions[agent] = model.predict(
            X_cal[features]
        ).astype(int)

        class_index = list(model.classes_).index(1)

        calibration_probabilities[agent] = model.predict_proba(
            X_cal[features]
        )[:, class_index]

        print(agent, "calibration predictions generated")

            # -----------------------------------------
    # ANALYZE AGENT CONFIDENCE CALIBRATION
    # -----------------------------------------

    print("\nANALYZING AGENT CONFIDENCE")
    print("=" * 50)

    calibration_summary = []
    calibration_bins = []

    y_cal_values = y_cal.to_numpy()

    for agent in agent_names:

        agent_predictions = calibration_predictions[agent]
        agent_probabilities = calibration_probabilities[agent]

        total_records = len(y_cal_values)

        correct_count = 0
        confidence_sum = 0.0
        ece = 0.0

        bin_data = []

        for position in range(total_records):

            prediction = int(agent_predictions[position])
            probability = float(agent_probabilities[position])

            confidence = (
                probability if prediction == 1
                else 1 - probability
            )

            correct = int(
                prediction == int(y_cal_values[position])
            )

            correct_count += correct
            confidence_sum += confidence

            bin_data.append({
                "confidence": confidence,
                "correct": correct
            })

        # Calculate Expected Calibration Error.
        for bin_number in range(10):

            lower = bin_number / 10
            upper = (bin_number + 1) / 10

            selected = [
                item for item in bin_data
                if item["confidence"] >= lower
                and (
                    item["confidence"] < upper
                    or (bin_number == 9
                        and item["confidence"] <= upper)
                )
            ]

            if not selected:
                continue

            bin_accuracy = sum(
                item["correct"] for item in selected
            ) / len(selected)

            bin_confidence = sum(
                item["confidence"] for item in selected
            ) / len(selected)

            bin_ece = (
                len(selected) / total_records
            ) * abs(bin_accuracy - bin_confidence)

            ece += bin_ece

            calibration_bins.append({
                "Agent": agent,
                "Confidence Range": f"{lower:.1f}-{upper:.1f}",
                "Records": len(selected),
                "Accuracy": round(bin_accuracy, 6),
                "Mean Confidence": round(bin_confidence, 6)
            })

        accuracy = correct_count / total_records

        brier = brier_score_loss(
            y_cal_values,
            agent_probabilities
        )

        mean_confidence = confidence_sum / total_records

        calibration_summary.append({
            "Agent": agent,
            "Accuracy": round(accuracy, 6),
            "Mean Confidence": round(mean_confidence, 6),
            "Brier Score": round(brier, 6),
            "ECE": round(ece, 6)
        })

        print(
            agent,
            "| Accuracy:", round(accuracy, 4),
            "| Brier:", round(brier, 4),
            "| ECE:", round(ece, 4)
        )

    os.makedirs("results", exist_ok=True)

    pd.DataFrame(calibration_summary).to_csv(
        "results/agent_confidence_calibration.csv",
        index=False
    )

    pd.DataFrame(calibration_bins).to_csv(
        "results/agent_confidence_bins.csv",
        index=False
    )

    print("\nCALIBRATION RESULTS SAVED")
    print("results/agent_confidence_calibration.csv")
    print("results/agent_confidence_bins.csv")

    # -----------------------------------------
    # TRAIN STATISTICAL EVIDENCE AGENT
    # -----------------------------------------

    print("\nTRAINING STATISTICAL EVIDENCE AGENT")
    print("=" * 50)

    evidence_agent = EvidenceAgent(EVIDENCE_PATH)

    reference_df = X_train.copy()
    reference_df["label"] = y_train

    evidence_agent.fit_reference(
        reference_df,
        target_column="label"
    )

    # -----------------------------------------
    # INITIALIZE CRITIC AND CORRECTION
    # -----------------------------------------

    critic = CriticAgent()

    correction = SelfCorrection()

    # Tune the override threshold using calibration data only.
    calibration_cases = []

    for position, (index, row) in enumerate(X_cal.iterrows()):

        opinions = {
            agent: int(calibration_predictions[agent][position])
            for agent in agent_names
        }

        critic_result = critic.evaluate(
            "UNSW-NB15 network traffic classification",
            opinions,
            []
        )

        majority = (
            1 if critic_result["malicious_votes"] > critic_result["normal_votes"]
            else 0 if critic_result["normal_votes"] > critic_result["malicious_votes"]
            else None
        )

        evidence = evidence_agent.generate_record_evidence(row)
        agent_evidence = evidence_agent.aggregate_agent_evidence(evidence)

        attack_agents = sum(
            result["decision"] == 1 for result in agent_evidence.values()
        )
        normal_agents = sum(
            result["decision"] == 0 for result in agent_evidence.values()
        )

        evidence_decision = None
        if attack_agents >= 3 and normal_agents == 0:
            evidence_decision = 1
        elif normal_agents >= 3 and attack_agents == 0:
            evidence_decision = 0

        reliabilities = []

        if majority is not None:
            for agent in agent_names:
                if opinions[agent] == majority:
                    p = float(calibration_probabilities[agent][position])
                    confidence = p if opinions[agent] == 1 else 1 - p
                    reliability = correction.get_calibrated_reliability(
                        agent, confidence
                    )
                    if reliability is not None:
                        reliabilities.append(reliability)

        avg_reliability = (
            sum(reliabilities) / len(reliabilities)
            if reliabilities else None
        )

        calibration_cases.append({
            "majority": majority,
            "evidence": evidence_decision,
            "reliability": avg_reliability,
            "actual": int(y_cal.loc[index])
        })

    # Test candidate thresholds on calibration data.
    thresholds = sorted({
        0.0, 1.0,
        *[
            case["reliability"]
            for case in calibration_cases
            if case["reliability"] is not None
        ]
    })

    best_threshold = 0.0
    best_f1 = -1

    for threshold in thresholds:

        actual = []
        predicted = []

        for case in calibration_cases:
            decision = case["majority"]

            if decision is None:
                decision = 0

            if (
                case["evidence"] is not None
                and case["majority"] is not None
                and case["evidence"] != case["majority"]
                and case["reliability"] is not None
                and case["reliability"] <= threshold
            ):
                decision = case["evidence"]

            actual.append(case["actual"])
            predicted.append(decision)

        score = f1_score(
            actual, predicted, average="macro", zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    correction.max_majority_reliability_for_override = best_threshold

    print("\nCALIBRATED CORRECTION THRESHOLD")
    print("Selected threshold:", round(best_threshold, 6))
    print("Calibration Macro F1:", round(best_f1, 6))

    records = []

    print("\nRUNNING SELF-CORRECTION EVALUATION")
    print("=" * 50)

    # -----------------------------------------
    # PROCESS EVALUATION RECORDS
    # -----------------------------------------

    for position, (index, row) in enumerate(X_eval.iterrows()):

        if (position + 1) % 50 == 0:

            print(
                f"Processed {position + 1}/{len(X_eval)} records"
            )

        # Actual specialist ML predictions.
        opinions = {
            agent: int(predictions[agent][position])
            for agent in agent_names
        }

        # Generate record-specific evidence.
        record_evidence = evidence_agent.generate_record_evidence(
            row
        )

        # Aggregate evidence for each specialist.
        agent_evidence = evidence_agent.aggregate_agent_evidence(
            record_evidence
        )

        # Critic evaluates actual agent opinions.
        critic_result = critic.evaluate(
            "UNSW-NB15 network traffic classification",
            opinions,
            []
        )

        # Run evidence-aware self-correction.
        # Calculate prediction confidence for every agent
        agent_confidences = {}

        for agent_name in agent_names:
            attack_probability = float(
                probabilities[agent_name][position]
            )
            agent_prediction = opinions[agent_name]

            prediction_confidence = (
                attack_probability
                if agent_prediction == 1
                else 1.0 - attack_probability
            )

            agent_confidences[agent_name] = prediction_confidence

        # Run confidence-aware self-correction
        correction_result = correction.correct(
            opinions=opinions,
            critic_result=critic_result,
            evidence=record_evidence,
            agent_evidence=agent_evidence,
            agent_confidences=agent_confidences
        )

        true_label = int(y_eval.loc[index])

        before = majority_vote(opinions)

        after = correction_result["final_decision"]

        if after is None:
            after = before

        changed = before != after

        # -----------------------------------------
        # SPECIALIST-LEVEL DIAGNOSTICS
        # -----------------------------------------

        agent_diagnostics = {}

        for agent_name, result in agent_evidence.items():

            safe_name = agent_name.lower().replace(" ", "_")

            agent_prediction = opinions[agent_name]

            # Actual ML probability of attack.
            attack_probability = float(
                probabilities[agent_name][position]
            )

            # Confidence in the predicted class.
            prediction_confidence = (
                attack_probability
                if agent_prediction == 1
                else 1 - attack_probability
            )

            agent_diagnostics[
                f"{safe_name}_prediction"
            ] = agent_prediction

            agent_diagnostics[
                f"{safe_name}_attack_probability"
            ] = round(attack_probability, 6)

            agent_diagnostics[
                f"{safe_name}_prediction_confidence"
            ] = round(prediction_confidence, 6)

            agent_diagnostics[
                f"{safe_name}_evidence_decision"
            ] = result["decision"]

            agent_diagnostics[
                f"{safe_name}_attack_strength"
            ] = result["attack_strength"]

            agent_diagnostics[
                f"{safe_name}_normal_strength"
            ] = result["normal_strength"]

            agent_diagnostics[
                f"{safe_name}_total_features"
            ] = result["total_features"]

        # -----------------------------------------
        # SAVE RECORD-LEVEL RESULTS
        # -----------------------------------------

        record = {

            "row_index": int(index),

            "true_label": true_label,

            "majority_before": before,

            "decision_after": int(after),

            "decision_changed": changed,

            "before_correct": before == true_label,

            "after_correct": after == true_label,

            "correction_success": (
                changed and after == true_label
            ),

            "false_correction": (
                changed
                and before == true_label
                and after != true_label
            ),

            "evidence_decision": correction_result[
                "evidence_decision"
            ],

            "evidence_items": len(record_evidence),

            "attack_evidence_agents": len(
                correction_result["attack_agents"]
            ),

            "normal_evidence_agents": len(
                correction_result["normal_agents"]
            ),

            "uncertain_evidence_agents": len(
                correction_result["uncertain_agents"]
            )
        }

        record.update(agent_diagnostics)

        records.append(record)

    # -----------------------------------------
    # SAVE RESULTS
    # -----------------------------------------

    results = pd.DataFrame(records)

    results.to_csv(
        RESULT_PATH,
        index=False
    )

    # -----------------------------------------
    # CALCULATE PERFORMANCE
    # -----------------------------------------

    y_true = results["true_label"]

    before_predictions = results["majority_before"]

    after_predictions = results["decision_after"]

    before_metrics = calculate_metrics(
        y_true,
        before_predictions
    )

    after_metrics = calculate_metrics(
        y_true,
        after_predictions
    )

    changed_count = int(
        results["decision_changed"].sum()
    )

    successful = int(
        results["correction_success"].sum()
    )

    false_corrections = int(
        results["false_correction"].sum()
    )

    total_records = len(results)

    summary = pd.DataFrame([

        {
            "Method": "Majority Before Correction",
            **before_metrics
        },

        {
            "Method": "Evidence-Based After Correction",
            **after_metrics
        }

    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # -----------------------------------------
    # FINAL REPORT
    # -----------------------------------------

    print("\nFINAL SELF-CORRECTION EVALUATION")
    print("=" * 55)

    print("\nBEFORE CORRECTION")

    print(
        "Accuracy:",
        round(before_metrics["Accuracy"], 4)
    )

    print(
        "Macro F1:",
        round(before_metrics["Macro F1"], 4)
    )

    print("\nAFTER CORRECTION")

    print(
        "Accuracy:",
        round(after_metrics["Accuracy"], 4)
    )

    print(
        "Macro F1:",
        round(after_metrics["Macro F1"], 4)
    )

    print("\nCORRECTION ANALYSIS")

    print("Total records:", total_records)

    print("Changed decisions:", changed_count)

    print("Successful corrections:", successful)

    print("False corrections:", false_corrections)

    print(
        "Correction coverage:",
        round(changed_count / total_records, 4)
    )

    print(
        "Successful correction rate:",
        round(
            successful / changed_count if changed_count else 0,
            4
        )
    )

    print(
        "False correction rate:",
        round(
            false_corrections / changed_count
            if changed_count else 0,
            4
        )
    )

    print("\nRESULTS SAVED")

    print(RESULT_PATH)

    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
