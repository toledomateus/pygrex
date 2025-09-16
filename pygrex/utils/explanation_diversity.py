from itertools import combinations
import numpy as np


def _get_explanation_feature_set(explanation, explainer_type, details=None):
    """Helper to extract a consistent feature set from different explanation types."""
    if explainer_type == "Sliding Window":
        return set(explanation.get("items", []))
    elif explainer_type == "EXPGRS":
        if details is not None:
            return set(details.get("antecedent", frozenset()))
        else:
            return set()
    elif explainer_type == "LORE4Groups":
        rules_data = explanation.get("group_factual_rule", {})
        if isinstance(rules_data, dict):
            return set(
                rule for tier_rules in rules_data.values() for rule in tier_rules
            )
        elif isinstance(rules_data, list):
            return set(rules_data)
    return set()


def calculate_gild_for_explanations(explanations_dict, explainer_type, use_median=True):
    """Calculate Gaussian Inter-List Diversity (GILD) for a set of explanations."""

    if not explanations_dict or len(explanations_dict) < 2:
        return 0.0

    feature_sets = []
    if explainer_type == "EXPGRS":
        for item_id, rules_list in explanations_dict.items():
            if rules_list:
                feature_sets.append(
                    _get_explanation_feature_set(
                        None, explainer_type, details=rules_list[0]
                    )
                )
    elif explainer_type == "Sliding Window":
        for call, exp_data in explanations_dict.items():
            feature_sets.append(_get_explanation_feature_set(exp_data, explainer_type))
    elif explainer_type == "LORE4Groups":
        for item_id, exp_data in explanations_dict.items():
            feature_sets.append(_get_explanation_feature_set(exp_data, explainer_type))

    feature_sets = [fs for fs in feature_sets if fs]
    if len(feature_sets) < 2:
        return 0.0

    # Calculate pairwise Jaccard distances
    distances = []
    for set1, set2 in combinations(feature_sets, 2):
        intersection_len = len(set1.intersection(set2))
        union_len = len(set1.union(set2))
        jaccard_dist = 1.0 - (intersection_len / union_len) if union_len > 0 else 1.0
        distances.append(jaccard_dist)

    if not distances:
        return 0.0

    # Calculate sigma using paper's formula
    k_choose_2 = len(distances)
    if use_median:
        reference_dist = np.median(distances)
    else:
        reference_dist = min(distances)

    denominator = np.sqrt(2 * np.log(k_choose_2 - 1)) if k_choose_2 > 1 else 1.0
    sigma = reference_dist / denominator if denominator > 0 else reference_dist
    if sigma == 0:
        sigma = 1e-9
    kernel_distances_sum = 0.0
    for d in distances:
        kernel_distance = np.sqrt(2 - 2 * np.exp(-(d**2) / (2 * sigma**2)))
        kernel_distances_sum += kernel_distance

    gild = kernel_distances_sum / k_choose_2 if distances else 0

    return gild
