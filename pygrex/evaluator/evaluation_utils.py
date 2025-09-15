import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, Tuple
import time

from pygrex.data_reader.data_reader import DataReader
from pygrex.evaluator import Splitter, Evaluator


def run_leave_one_out_evaluation_improved(
    data_reader: DataReader, model, top_n: int = 10
) -> Dict:
    """
    Improved leave-one-out evaluation using the existing Evaluator class.
    Much more efficient and properly implemented.
    """
    print("Starting leave-one-out evaluation...")
    start_time = time.time()

    # 1. Proper leave-one-out split (one item per user)
    train_dr, test_df = Splitter.split_leave_n_out(
        data_reader, n=1
    )  # n=1 for true leave-one-out
    print(f"Split completed: {len(test_df)} test interactions")

    # 2. Train model on training data
    print("Training model on reduced dataset...")
    train_start = time.time()
    model.fit(train_dr)
    train_time = time.time() - train_start
    print(f"Model training completed in {train_time:.2f} seconds")

    # 3. Generate recommendations efficiently
    print("Generating recommendations...")
    rec_start = time.time()
    recommendations = generate_recommendations_batch(model, train_dr, test_df, top_n)
    rec_time = time.time() - rec_start
    print(f"Recommendations generated in {rec_time:.2f} seconds")

    # 4. Use the existing Evaluator class
    evaluator = Evaluator(test_df, top_n=top_n)

    # Calculate metrics
    hit_ratio = evaluator.cal_hit_ratio(recommendations)
    ndcg = evaluator.cal_ndcg(recommendations)

    total_time = time.time() - start_time
    print(f"Total evaluation time: {total_time:.2f} seconds")

    return {
        "Hit Ratio": hit_ratio,
        "NDCG": ndcg,  # Using standard NDCG instead of eNDCG for now
        "evaluation_time": total_time,
    }


def generate_recommendations_batch(
    model, train_dr: DataReader, test_df: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """
    Generate recommendations in batch mode for efficiency.
    Returns DataFrame with columns: ['userId', 'itemId', 'rank', 'score']
    """
    all_items = set(train_dr.dataset["itemId"].unique())
    recommendations = []

    test_users = test_df["userId"].unique()
    print(f"Generating recommendations for {len(test_users)} users...")

    for i, user_id in enumerate(test_users):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing user {i}/{len(test_users)}")

        # Get items the user has already interacted with
        user_items = set(
            train_dr.dataset[train_dr.dataset["userId"] == user_id]["itemId"]
        )

        # Candidate items (unseen items)
        candidate_items = list(all_items - user_items)

        # For efficiency, limit candidates if there are too many
        if len(candidate_items) > 10000:  # Adjust this threshold based on your needs
            candidate_items = np.random.choice(
                candidate_items, 10000, replace=False
            ).tolist()

        # Generate predictions - try to use batch prediction if available
        try:
            # Check if model has batch prediction capability
            if hasattr(model, "predict_batch") or hasattr(model, "recommend"):
                user_recs = generate_recommendations_efficient(
                    model, user_id, candidate_items, top_n
                )
            else:
                # Fall back to individual predictions (slower)
                user_recs = generate_recommendations_individual(
                    model, user_id, candidate_items, top_n
                )

            recommendations.extend(user_recs)

        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue

    # Convert to DataFrame
    if recommendations:
        rec_df = pd.DataFrame(
            recommendations, columns=["userId", "itemId", "rank", "score"]
        )
    else:
        # Return empty DataFrame with correct structure
        rec_df = pd.DataFrame(columns=["userId", "itemId", "rank", "score"])

    return rec_df


def generate_recommendations_efficient(
    model, user_id: int, candidate_items: list, top_n: int
) -> list:
    """
    Try to use efficient recommendation methods if available.
    """
    recommendations = []

    # Try different efficient methods based on model type
    if hasattr(model, "recommend"):
        # Some models have a recommend method
        try:
            recs = model.recommend(user_id, candidate_items, top_n)
            for rank, (item_id, score) in enumerate(recs, 1):
                recommendations.append((user_id, item_id, rank, score))
        except:
            # Fall back to individual predictions
            return generate_recommendations_individual(
                model, user_id, candidate_items, top_n
            )

    elif hasattr(model, "predict_batch"):
        # Batch prediction if available
        try:
            user_items_batch = [(user_id, item_id) for item_id in candidate_items]
            scores = model.predict_batch(user_items_batch)

            # Sort by score and get top-N
            scored_items = list(zip(candidate_items, scores))
            scored_items.sort(key=lambda x: x[1], reverse=True)

            for rank, (item_id, score) in enumerate(scored_items[:top_n], 1):
                recommendations.append((user_id, item_id, rank, score))
        except:
            return generate_recommendations_individual(
                model, user_id, candidate_items, top_n
            )

    else:
        return generate_recommendations_individual(
            model, user_id, candidate_items, top_n
        )

    return recommendations


def generate_recommendations_individual(
    model, user_id: int, candidate_items: list, top_n: int
) -> list:
    """
    Fall back to individual predictions (slower but works with any model).
    """
    predictions = []

    # Batch the individual predictions for better performance
    batch_size = 100
    for i in range(0, len(candidate_items), batch_size):
        batch_items = candidate_items[i : i + batch_size]

        for item_id in batch_items:
            try:
                score = model.predict(user_id, item_id)
                predictions.append((item_id, score))
            except Exception as e:
                print(f"Prediction error for user {user_id}, item {item_id}: {e}")
                # Skip items that cause prediction errors
                continue

    # Sort by score and get top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_predictions = predictions[:top_n]

    recommendations = []
    for rank, (item_id, score) in enumerate(top_predictions, 1):
        recommendations.append((user_id, item_id, rank, score))

    return recommendations


def run_evaluation_with_proper_split(
    data_reader: DataReader, model, test_size: float = 0.2, top_n: int = 10
) -> Dict:
    """
    Alternative evaluation using a proper train/test split instead of leave-one-out.
    This is often more practical and much faster.
    """
    print(f"Starting evaluation with {test_size * 100}% test split...")
    start_time = time.time()

    # 1. Split data into train/test
    train_dr, test_df = Splitter.split_leave_n_out(data_reader, frac=test_size)
    print(f"Split completed: {len(test_df)} test interactions")

    # 2. Train model
    print("Training model...")
    model.fit(train_dr)

    # 3. Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations_batch(model, train_dr, test_df, top_n)

    # 4. Evaluate
    evaluator = Evaluator(test_df, top_n=top_n)
    hit_ratio = evaluator.cal_hit_ratio(recommendations)
    ndcg = evaluator.cal_ndcg(recommendations)

    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f} seconds")

    return {
        "Hit Ratio": hit_ratio,
        "NDCG": ndcg,
        "evaluation_time": total_time,
        "test_interactions": len(test_df),
        "total_recommendations": len(recommendations),
    }


# Gaussian ILD (GILD) - Your existing implementation looks good
def _get_explanation_feature_set(explanation, explainer_type, details=None):
    """Helper to extract a consistent feature set from different explanation types."""
    if explainer_type == "Sliding Window":
        return set(explanation.get("items", []))
    elif explainer_type == "EXPGRS":
        return set(details.get("antecedent", frozenset()))
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
