
from influence.influence_graph import DynamicInfluenceGraph
from consensus.consensus_engine import ConsensusEngine


def main():

    influence_graph = DynamicInfluenceGraph()
    engine = ConsensusEngine(influence_graph)

    opinions = {
        "Traffic Agent": 1,
        "Threat Agent": 1,
        "Behaviour Agent": 0,
        "Risk Agent": 0
    }

    # TEST 1: BEFORE RELIABILITY UPDATE

    print("\nBEFORE RELIABILITY UPDATE")
    print("=" * 45)

    before_weights = engine.calculate_dynamic_weights(opinions)

    for agent, weight in before_weights.items():
        print(f"{agent}: {weight:.4f}")

    print(
        "Total weight:",
        round(sum(before_weights.values()), 4)
    )

    before_decision = engine.proposed_consensus(opinions)

    print("Consensus decision:", before_decision)

    # TEST 2: OVERALL RELIABILITY UPDATE

    print("\nUPDATING TRAFFIC AGENT RELIABILITY")

    influence_graph.update_reliability(
        "Traffic Agent",
        correct=30,
        total=100
    )

    print("\nAFTER RELIABILITY UPDATE")
    print("=" * 45)

    after_weights = engine.calculate_dynamic_weights(opinions)

    for agent, weight in after_weights.items():
        print(f"{agent}: {weight:.4f}")

    print(
        "Total weight:",
        round(sum(after_weights.values()), 4)
    )

    after_decision = engine.proposed_consensus(opinions)

    print("Consensus decision:", after_decision)

    # TEST 3: CONTEXT-AWARE RELIABILITY

    print("\nTESTING CONTEXT-AWARE RELIABILITY")
    print("=" * 50)

    influence_graph.update_context_reliability(
        agent_name="Traffic Agent",
        normal_correct=80,
        normal_total=100,
        attack_correct=95,
        attack_total=100
    )

    print("\nTraffic Agent reliability:")

    print(
        "Overall:",
        round(
            influence_graph.graph.nodes["Traffic Agent"]["reliability"],
            4
        )
    )

    print(
        "Normal:",
        round(
            influence_graph.get_context_reliability(
                "Traffic Agent", 0
            ), 4
        )
    )

    print(
        "Attack:",
        round(
            influence_graph.get_context_reliability(
                "Traffic Agent", 1
            ), 4
        )
    )

    print("\nCOMPLETE INFLUENCE GRAPH")
    influence_graph.display_graph()


if __name__ == "__main__":
    main()
