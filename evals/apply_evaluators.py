import json
import networkx as nx
from networkx.algorithms import graph_edit_distance
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from langsmith.schemas import Example, Run
from langsmith.beta import compute_test_metrics

def state_schema_match(run: Run, example: Example) -> dict:
    predicted_graph: dict = run.outputs
    expected_graph: dict = example.outputs

    # Parse the JSON strings into dictionaries

    # Extract the state schemas from the predicted and expected graphs
    predicted_state = predicted_graph["state"]
    expected_state = expected_graph["state"]
    # Recursively compare the properties and annotations
    def compare_state(pred_state, exp_state):
        if isinstance(pred_state, dict) and isinstance(exp_state, dict):
            pred_keys = set(pred_state.keys())
            exp_keys = set(exp_state.keys())
            common_keys = pred_keys.intersection(exp_keys)

            total_keys = len(pred_keys) + len(exp_keys)
            matching_keys = sum(
                compare_state(pred_state[key], exp_state[key]) for key in common_keys
            )

            return matching_keys / total_keys

        return 1.0 if pred_state == exp_state else 0.0

    state_match_percentage = compare_state(predicted_state, expected_state)

    return {"score": state_match_percentage}


def node_edge_correctness(run: Run, example: Example) -> dict:
    predicted_graph: dict = run.outputs
    expected_graph: dict = example.outputs

    # Parse the JSON strings into dictionaries

    # Extract nodes and edges from the predicted and expected graphs
    predicted_nodes = predicted_graph["nodes"]
    expected_nodes = expected_graph["nodes"]
    predicted_edges = predicted_graph["edges"]
    expected_edges = expected_graph["edges"]

    # Compare nodes
    node_matches = [node in expected_nodes for node in predicted_nodes]
    node_precision = precision_score(node_matches, [True] * len(node_matches))
    node_recall = recall_score(node_matches, [True] * len(node_matches))
    node_f1 = f1_score(node_matches, [True] * len(node_matches))

    # Compare edges
    edge_matches = [edge in expected_edges for edge in predicted_edges]
    edge_precision = precision_score(edge_matches, [True] * len(edge_matches))
    edge_recall = recall_score(edge_matches, [True] * len(edge_matches))
    edge_f1 = f1_score(edge_matches, [True] * len(edge_matches))
    results = [
        {"key": "node_precision", "value": node_precision},
        {"key": "node_recall", "value": node_recall},
        {"key": "node_f1", "value": node_f1},
        {"key": "edge_precision", "value": edge_precision},
        {"key": "edge_recall", "value": edge_recall},
        {"key": "edge_f1", "value": edge_f1},
    ]
    return {"score": results}


def graph_structure_similarity(run: Run, example: Example) -> dict:
    predicted_graph: dict = run.outputs
    expected_graph: dict = example.outputs

    # Parse the JSON strings into dictionaries

    # Convert the dictionaries into NetworkX graphs
    G_predicted = nx.from_dict_of_lists(predicted_graph)
    G_expected = nx.from_dict_of_lists(expected_graph)

    # Calculate the graph edit distance
    edit_distance = graph_edit_distance(G_predicted, G_expected)

    # Normalize the edit distance
    max_distance = max(
        len(G_predicted.nodes) + len(G_expected.nodes),
        len(G_predicted.edges) + len(G_expected.edges),
    )
    similarity_score = 1 - (edit_distance / max_distance)

    return {"score": similarity_score}


compute_test_metrics("clear-decision-77", evaluators=[state_schema_match, node_edge_correctness, graph_structure_similarity])
