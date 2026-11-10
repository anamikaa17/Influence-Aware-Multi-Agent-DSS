
import json
import os


class SelfCorrection:

    def __init__(
        self,
        reliability_path="config/self_correction_reliability.json"
    ):
        self.name = "Confidence-Aware Self-Correction Engine"

        self.agent_expertise = {
            "Traffic Agent": [
                "traffic", "packet", "volume", "connection"
            ],
            "Threat Agent": [
                "attack", "malicious", "intrusion",
                "brute force", "suspicious"
            ],
            "Behaviour Agent": [
                "behaviour", "pattern", "anomalous", "unusual"
            ],
            "Risk Agent": [
                "risk", "security", "decision", "external"
            ]
        }

        self.reliability = {}

        if os.path.exists(reliability_path):
            with open(reliability_path, "r") as file:
                self.reliability = json.load(file)

        else:
            raise FileNotFoundError(
                f"Reliability profile not found: {reliability_path}"
            )

        # Conservative provisional threshold.
        # This must be selected using calibration data.
        self.max_majority_reliability_for_override = 0.75

    def assess_evidence(self, agent_name, evidence):

        keywords = self.agent_expertise.get(agent_name, [])

        return [
            item for item in evidence
            if any(
                word in item.get("evidence", "").lower()
                for word in keywords
            )
        ]

    def get_calibrated_reliability(
        self,
        agent_name,
        confidence
    ):

        agent_profile = self.reliability.get(agent_name, {})

        bins = agent_profile.get("confidence_bins", {})

        if not bins:
            return None

        confidence = max(0.0, min(1.0, float(confidence)))

        if confidence < 0.60:
            bin_name = "0.5-0.6"
        elif confidence < 0.70:
            bin_name = "0.6-0.7"
        elif confidence < 0.80:
            bin_name = "0.7-0.8"
        elif confidence < 0.90:
            bin_name = "0.8-0.9"
        else:
            bin_name = "0.9-1.0"

        bin_data = bins.get(bin_name)

        if bin_data is None:
            return None

        return float(bin_data["accuracy"])

    def correct(
        self,
        opinions,
        critic_result,
        evidence,
        agent_evidence=None,
        agent_confidences=None
    ):

        print("\nCONFIDENCE-AWARE SELF-CORRECTION")
        print("=" * 50)

        if not opinions:
            return {
                "original_opinions": {},
                "corrected_opinions": {},
                "majority_decision": None,
                "evidence_decision": None,
                "final_decision": None,
                "correction_log": ["No opinions available."]
            }

        corrected_opinions = opinions.copy()
        correction_log = []

        malicious_votes = critic_result["malicious_votes"]
        normal_votes = critic_result["normal_votes"]

        majority_decision = (
            1 if malicious_votes > normal_votes
            else 0 if normal_votes > malicious_votes
            else None
        )

        # ----------------------------------
        # EVALUATE SPECIALIST EVIDENCE
        # ----------------------------------

        attack_agents = []
        normal_agents = []
        uncertain_agents = []

        if agent_evidence is not None:

            for agent_name, result in agent_evidence.items():

                decision = result["decision"]

                if decision == 1:
                    attack_agents.append(agent_name)

                elif decision == 0:
                    normal_agents.append(agent_name)

                else:
                    uncertain_agents.append(agent_name)

        print("\nSPECIALIST EVIDENCE ASSESSMENT")
        print("Attack-supporting agents:", attack_agents)
        print("Normal-supporting agents:", normal_agents)
        print("Uncertain agents:", uncertain_agents)

        # ----------------------------------
        # ESTIMATE MAJORITY RELIABILITY
        # ----------------------------------

        majority_reliability = []

        if majority_decision is not None:

            for agent_name, prediction in opinions.items():

                if prediction != majority_decision:
                    continue

                confidence = None

                if agent_confidences is not None:
                    confidence = agent_confidences.get(agent_name)

                if confidence is None:
                    continue

                reliability = self.get_calibrated_reliability(
                    agent_name,
                    confidence
                )

                if reliability is not None:
                    majority_reliability.append(reliability)

        average_majority_reliability = (
            sum(majority_reliability) / len(majority_reliability)
            if majority_reliability else None
        )

        print("\nCONFIDENCE RELIABILITY")

        print(
            "Average majority reliability:",
            average_majority_reliability
        )

        # ----------------------------------
        # DETERMINE EVIDENCE DECISION
        # ----------------------------------

        evidence_decision = None

        if len(attack_agents) >= 3 and len(normal_agents) == 0:
            evidence_decision = 1

        elif len(normal_agents) >= 3 and len(attack_agents) == 0:
            evidence_decision = 0

        print("\nDECISION ANALYSIS")

        print("Majority decision:", majority_decision)
        print("Evidence decision:", evidence_decision)

        # ----------------------------------
        # CONFIDENCE-AWARE FINAL DECISION
        # ----------------------------------

        final_decision = majority_decision

        if evidence_decision is None:

            correction_log.append(
                "Insufficient specialist evidence. "
                "Original majority retained."
            )

        elif majority_decision is None:

            correction_log.append(
                "ML agents are tied. Evidence is recorded "
                "as provisional; manual verification required."
            )

        elif evidence_decision == majority_decision:

            correction_log.append(
                "Specialist evidence supports the ML majority."
            )

        elif average_majority_reliability is None:

            correction_log.append(
                "Majority reliability unavailable. "
                "Original majority retained."
            )

        elif (
            average_majority_reliability
            <= self.max_majority_reliability_for_override
             and not any(
             prediction == majority_decision
             and agent_confidences is not None
             and agent_confidences.get(agent_name, 0) >= 0.80
             for agent_name, prediction in opinions.items()
    )
):

            final_decision = evidence_decision

            correction_log.append(
                "Evidence challenged a low-reliability majority. "
                "Provisional correction applied."
            )

        else:

            correction_log.append(
                "Evidence conflicts with a reliable ML majority. "
                "Original majority retained."
            )

        # ----------------------------------
        # FINAL REPORT
        # ----------------------------------

        print("\nFINAL ASSESSMENT")

        if final_decision is None:
            print("Manual verification required.")
        else:
            print(
                "Final decision:",
                "MALICIOUS" if final_decision == 1 else "NORMAL"
            )

        print("\nCORRECTION SUMMARY")

        for message in correction_log:
            print("-", message)

        return {
            "original_opinions": opinions.copy(),
            "corrected_opinions": corrected_opinions,
            "majority_decision": majority_decision,
            "evidence_decision": evidence_decision,
            "final_decision": final_decision,
            "attack_agents": attack_agents,
            "normal_agents": normal_agents,
            "uncertain_agents": uncertain_agents,
            "average_majority_reliability": (
                average_majority_reliability
            ),
            "correction_log": correction_log
        }
