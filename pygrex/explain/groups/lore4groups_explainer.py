import pandas as pd
import numpy as np
import re
import logging
import traceback
from collections import Counter
from typing import Dict, Set, List, Optional, Any, Tuple, Union
from sklearn.tree import DecisionTreeClassifier, _tree

ItemId = Union[str, int]
UserId = Union[str, int]
FactualRule = List[str]
CounterfactualSet = List[List[str]]
Explanation = Tuple[Optional[FactualRule], Optional[CounterfactualSet]]


class LORE4GroupsExplainer:
    """
    Enhanced LORE4Groups explainer that incorporates genre information
    and stores decision trees for visualization
    """

    def __init__(
        self,
        item_profiles: Dict[str, Set[str]],
        item_label_matrix: pd.DataFrame,
        config: Dict,
        genre_profiles: Optional[Dict[str, Set[str]]] = None,
    ):
        self.item_profiles = {str(k): v for k, v in item_profiles.items()}
        self.item_label_matrix = item_label_matrix
        self.params = config["explainer"]["lore4groups"]

        # NEW: Store genre information
        self.genre_profiles = (
            {str(k): v for k, v in genre_profiles.items()} if genre_profiles else {}
        )

        all_columns = item_label_matrix.columns.tolist()
        self.all_labels = [col for col in all_columns if col != "like"]

        # Add 'like' back for target variable access (but not as feature)
        if "like" in all_columns:
            self.all_labels.append("like")

    def _enhanced_jaccard_similarity(self, item1_id: ItemId, item2_id: ItemId) -> float:
        """Enhanced Jaccard similarity that considers both tags and genres"""
        # Get regular tags
        tags1 = self.item_profiles.get(str(item1_id), set())
        tags2 = self.item_profiles.get(str(item2_id), set())

        # Get genres and add them as features
        genres1 = self.genre_profiles.get(str(item1_id), set())
        genres2 = self.genre_profiles.get(str(item2_id), set())

        # Combine tags and genres for enhanced similarity
        features1 = tags1.union({f"genre_{g.lower()}" for g in genres1})
        features2 = tags2.union({f"genre_{g.lower()}" for g in genres2})

        if not features1 or not features2:
            return 0.0

        union_len = len(features1.union(features2))
        intersection_len = len(features1.intersection(features2))

        return intersection_len / union_len if union_len > 0 else 0.0

    def _jaccard_similarity(self, item1_id: ItemId, item2_id: ItemId) -> float:
        """Original jaccard similarity (kept for compatibility)"""
        tags1 = self.item_profiles.get(str(item1_id), set())
        tags2 = self.item_profiles.get(str(item2_id), set())
        if not tags1 or not tags2:
            return 0.0
        union_len = len(tags1.union(tags2))
        return len(tags1.intersection(tags2)) / union_len if union_len > 0 else 0.0

    def _get_enhanced_similar_examples(
        self,
        user_id_consecutive: UserId,
        target_item_id: ItemId,
        user_hist: Set[ItemId],
        dataset: pd.DataFrame,
        model=None,
        data_reader=None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced version that returns both DataFrame and metadata for visualization"""

        # 1. Find all similar items using enhanced similarity
        similarities = [
            (seen_id, self._enhanced_jaccard_similarity(target_item_id, seen_id))
            for seen_id in user_hist
        ]
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        sim_th = self.params.get("similarity_threshold", 0.0)
        top_similar_items_str = {
            item[0]
            for item in similarities[: self.params["n_similar_for_tree"]]
            if item[1] >= sim_th
        }

        if not top_similar_items_str:
            return pd.DataFrame(), {}

        # 2. Build the local dataset
        top_similar_items_int = [int(i) for i in top_similar_items_str]

        # Get existing ratings for similar items
        local_df = dataset[
            (dataset["userId"] == user_id_consecutive)
            & (dataset["itemId"].isin(top_similar_items_int))
        ].copy()

        rated_items = set(local_df["itemId"])
        items_to_predict = [
            item for item in top_similar_items_int if item not in rated_items
        ]

        # Add predictions for unrated items
        if model and data_reader and items_to_predict:
            try:
                orig_user_id = data_reader.get_original_user_id(
                    int(user_id_consecutive)
                )
                predicted_ratings = []

                for item_id_consecutive in items_to_predict:
                    orig_item_id = data_reader.get_original_item_id(
                        int(item_id_consecutive)
                    )
                    pred = model.predict(orig_user_id, orig_item_id)
                    predicted_ratings.append(
                        {
                            "userId": user_id_consecutive,
                            "itemId": item_id_consecutive,
                            "rating": float(pred),
                        }
                    )

                if predicted_ratings:
                    pred_df = pd.DataFrame(predicted_ratings)
                    local_df = pd.concat([local_df, pred_df], ignore_index=True)

            except Exception:
                traceback.print_exc()

        # Check minimum samples requirement
        if len(local_df) < 2:
            return pd.DataFrame(), {}

        # 3. Apply thresholding with fallbacks
        rating_threshold = self.params["rating_threshold_for_like"]

        threshold_info = {
            "was_overridden": False,
            "original_threshold": rating_threshold,
            "final_threshold": rating_threshold,
        }

        local_df["like"] = (local_df["rating"] >= rating_threshold).astype(int)

        # Apply fallback thresholds if needed
        like_counts = local_df["like"].value_counts()

        if len(like_counts) < 2:
            # Try mean-based threshold
            mean_rating = local_df["rating"].mean()
            local_df["like"] = (local_df["rating"] >= mean_rating).astype(int)
            threshold_info["was_overridden"] = True
            threshold_info["final_threshold"] = mean_rating
            like_counts = local_df["like"].value_counts()
            if len(like_counts) < 2:
                return pd.DataFrame(), {}

        # Check for severe imbalance (>90% one class)
        min_class_ratio = like_counts.min() / len(local_df)
        if min_class_ratio < 0.1:
            if like_counts.min() < 2:
                return pd.DataFrame(), {}

        # 4. Construct the enhanced feature matrix (including genres)
        feature_labels = [label for label in self.all_labels if label != "like"]

        examples = []
        genre_features_used = set()

        for idx, row in local_df.iterrows():
            item_id = str(int(row["itemId"]))
            tags = self.item_profiles.get(item_id, set())
            genres = self.genre_profiles.get(item_id, set())

            # Create base example with target variables
            example = {
                "movie_id": item_id,
                "rating": row["rating"],
                "like": int(row["like"]),
            }

            # Add tag features (excluding 'like')
            for label in feature_labels:
                example[label] = 1 if label in tags else 0

            # Add genre features dynamically
            for genre in genres:
                genre_feature = f"genre_{genre.lower()}"
                example[genre_feature] = 1
                genre_features_used.add(genre_feature)

                # Also add to feature_labels if not already there
                if genre_feature not in feature_labels:
                    feature_labels.append(genre_feature)

            examples.append(example)

        # Ensure all examples have all genre features
        for example in examples:
            for genre_feature in genre_features_used:
                if genre_feature not in example:
                    example[genre_feature] = 0

        final_df = pd.DataFrame(examples)

        # Final validation
        if final_df["like"].nunique() < 2:
            return pd.DataFrame(), {}

        # Prepare metadata for visualization
        metadata = {
            "feature_labels": [label for label in feature_labels if label != "like"],
            "genre_features": list(genre_features_used),
            "similarity_scores": dict(similarities[:5]),  # Top 5 similarities
            "target_item_genres": self.genre_profiles.get(str(target_item_id), set()),
            "rating_threshold": threshold_info["final_threshold"],
            "threshold_info": threshold_info,
        }

        return final_df, metadata

    def _get_factual_path_for_item(
        self,
        clf: DecisionTreeClassifier,
        x_item: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        Traces the specific path an item takes through the decision tree
        and returns the corresponding factual rule set.
        """
        feature_labels = metadata.get("feature_labels", [])
        if not feature_labels:
            return None

        # 1. Get the sequence of nodes the item travels through
        node_indicator = clf.decision_path(x_item)
        node_index = node_indicator.indices[  # type: ignore
            node_indicator.indptr[0] : node_indicator.indptr[  # type: ignore
                1
            ]
        ]

        rules = []
        tree = clf.tree_

        # 2. Iterate through the path to build the rules
        # We stop at the second to last node because the last one is the leaf
        for i in range(len(node_index) - 1):
            node_id = node_index[i]
            child_node_id = node_index[i + 1]

            # Ensure this is not a leaf node
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:  # type: ignore
                feature_name = feature_labels[tree.feature[node_id]]  # type: ignore
                threshold = tree.threshold[node_id]  # type: ignore

                # 3. Determine if the path went left or right to form the rule
                if child_node_id == tree.children_left[node_id]:  # type: ignore
                    # Path went left (True condition for <= threshold)
                    rule = f"{feature_name} <= {threshold:.2f}"
                else:
                    # Path went right (False condition for <= threshold)
                    rule = f"{feature_name} > {threshold:.2f}"

                # Use the same enhanced formatting as before for consistency
                if feature_name.startswith("genre_"):
                    genre_name = feature_name.replace("genre_", "").title()
                    if child_node_id == tree.children_left[node_id]:  # type: ignore
                        rules.append(f"Does NOT have genre: `{genre_name}`")
                    else:
                        rules.append(f"Has genre: `{genre_name}`")
                else:
                    rules.append(rule)

        return rules if rules else None

    def _train_enhanced_decision_tree(
        self,
        user_id_consecutive: UserId,
        item_id: ItemId,
        user_hist: Set[ItemId],
        dataset: pd.DataFrame,
        model=None,
        data_reader=None,
    ) -> Tuple[Optional[DecisionTreeClassifier], Dict[str, Any]]:
        """Enhanced tree training that returns both classifier and metadata"""

        df_examples, metadata = self._get_enhanced_similar_examples(
            user_id_consecutive, item_id, user_hist, dataset, model, data_reader
        )

        if df_examples.empty:
            return None, {}

        like_counts = df_examples["like"].value_counts()

        if len(like_counts) < 2 or like_counts.min() < 2:
            return None, {}

        feature_labels = metadata.get("feature_labels", [])
        X = df_examples[feature_labels]
        y = df_examples["like"]

        # Verify feature matrix has variance
        feature_variances = X.var()
        if (feature_variances == 0).all():
            return None, {}

        clf = DecisionTreeClassifier(
            max_depth=5,  # Slightly deeper to accommodate genre features
            min_samples_split=max(4, len(df_examples) // 4),
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )

        try:
            clf.fit(X, y)

            # Enhanced feature importance analysis
            feature_importance = list(zip(feature_labels, clf.feature_importances_))
            important_features = [
                (f, imp) for f, imp in feature_importance if imp > 0.001
            ]
            genre_important_features = [
                (f, imp) for f, imp in important_features if f.startswith("genre_")
            ]

            # Add classifier and feature info to metadata
            metadata.update(
                {
                    "classifier": clf,
                    "feature_importance": dict(feature_importance),
                    "important_features": important_features,
                    "genre_important_features": genre_important_features,
                    "training_data_size": len(df_examples),
                    "class_distribution": like_counts.to_dict(),
                }
            )

            return clf, metadata

        except Exception as _:
            return None, {}

    def _get_enhanced_explanation_path(
        self,
        clf: DecisionTreeClassifier,
        x_item: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Enhanced explanation path that provides better rule descriptions"""

        if 1 not in clf.classes_:
            return None

        leaf_id = clf.apply(x_item)[0]  # type: ignore
        class_index = np.where(clf.classes_ == 1)[0]
        if not class_index.size or clf.tree_.value[leaf_id][0][class_index[0]] == 0:  # type: ignore
            return None

        node_indicator = clf.decision_path(x_item)
        node_index = node_indicator.indices[  # type: ignore
            node_indicator.indptr[0] : node_indicator.indptr[  # type: ignore
                1
            ]
        ]

        rules = []
        feature_labels = metadata.get("feature_labels", [])

        for i in range(len(node_index) - 1):  # Exclude leaf node
            node_id = node_index[i]
            next_node_id = node_index[i + 1]

            if clf.tree_.feature[node_id] != _tree.TREE_UNDEFINED:  # type: ignore
                feature_name = feature_labels[clf.tree_.feature[node_id]]  # type: ignore
                threshold = clf.tree_.threshold[node_id]  # type: ignore

                # Enhanced rule formatting based on feature type
                if feature_name.startswith("genre_"):
                    genre_name = feature_name.replace("genre_", "").title()
                    if next_node_id == clf.tree_.children_left[node_id]:  # type: ignore
                        rules.append(f"Does NOT have genre: `{genre_name}`")
                    else:
                        rules.append(f"Has genre: `{genre_name}`")
                else:
                    # Regular tag features
                    if next_node_id == clf.tree_.children_left[node_id]:  # type: ignore
                        rules.append(f"{feature_name} <= {threshold}")
                    else:
                        rules.append(f"{feature_name} > {threshold}")

        return rules

    def _generate_enhanced_individual_explanation(
        self, clf: DecisionTreeClassifier, item_id: ItemId, metadata: Dict[str, Any]
    ) -> Optional[Explanation]:
        """Enhanced individual explanation generation"""

        if str(item_id) not in self.item_label_matrix.index:
            return None

        x_item_full = self.item_label_matrix.loc[[str(item_id)]]
        feature_labels = metadata.get("feature_labels", [])

        try:
            # For genre features, we need to dynamically add them to the item
            item_genres = self.genre_profiles.get(str(item_id), set())

            # Create enhanced item representation
            enhanced_item_data = x_item_full.copy()

            # Add genre features
            for genre in item_genres:
                genre_feature = f"genre_{genre.lower()}"
                if genre_feature in feature_labels:
                    enhanced_item_data[genre_feature] = 1

            # Ensure all genre features exist (set to 0 if not present)
            for feature in feature_labels:
                if (
                    feature.startswith("genre_")
                    and feature not in enhanced_item_data.columns
                ):
                    enhanced_item_data[feature] = 0

            # Select only the features used in training
            x_item = enhanced_item_data[feature_labels]

        except KeyError as _:
            return None
        # Get enhanced factual rule
        # factual_rule = self._get_enhanced_explanation_path(clf, x_item, metadata)
        factual_rule = self._get_factual_path_for_item(clf, x_item, metadata)

        if not factual_rule:
            return None

        # Get counterfactuals (reuse existing method)
        counterfactual_set = self._get_counterfactual_paths(clf, x_item)
        if not counterfactual_set:
            return None

        return (factual_rule, counterfactual_set)

    def _get_counterfactual_paths(
        self, clf: DecisionTreeClassifier, x_item: pd.DataFrame
    ) -> Optional[CounterfactualSet]:
        """Original counterfactual path method (kept for compatibility)"""
        tree = clf.tree_
        paths = []

        def find_paths(node_id, current_path):
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:  # type: ignore
                class_index = np.where(clf.classes_ == 0)[0]
                if class_index.size and tree.value[node_id][0][class_index[0]] > 0:
                    paths.append(list(current_path))
                return
            feature_idx = tree.feature[node_id]  # type: ignore
            threshold = tree.threshold[node_id]  # type: ignore
            current_path.append((feature_idx, "<=", threshold))
            find_paths(tree.children_left[node_id], current_path)  # type: ignore
            current_path.pop()
            current_path.append((feature_idx, ">", threshold))
            find_paths(tree.children_right[node_id], current_path)  # type: ignore
            current_path.pop()

        find_paths(0, [])
        if not paths:
            return None

        min_nf = float("inf")
        counterfactuals = []
        for path in paths:
            nf = 0
            for feature_idx, op, threshold in path:
                if feature_idx < len(x_item.columns):
                    item_val = x_item.iloc[0, feature_idx]
                    if not (
                        (op == "<=" and item_val <= threshold)
                        or (op == ">" and item_val > threshold)
                    ):
                        nf += 1
            if nf < min_nf:
                min_nf = nf
                counterfactuals = [path]
            elif nf == min_nf:
                counterfactuals.append(path)

        # Enhanced counterfactual formatting
        formatted_counterfactuals = []
        for cf_path in counterfactuals:
            formatted_path = []
            for idx, op, _ in cf_path:
                if idx < len(x_item.columns):
                    feature_name = x_item.columns[idx]
                    if feature_name.startswith("genre_"):
                        genre_name = feature_name.replace("genre_", "").title()
                        if op == "<=":
                            formatted_path.append(
                                f"Does NOT have genre: `{genre_name}`"
                            )
                        else:
                            formatted_path.append(f"Has genre: `{genre_name}`")
                    else:
                        formatted_path.append(f"{feature_name} {op} 0.5")
            if formatted_path:
                formatted_counterfactuals.append(formatted_path)

        return formatted_counterfactuals if formatted_counterfactuals else None

    def _aggregate_factual_rules(
        self, individual_explanations: Dict[UserId, List[str]], total_group_size: int
    ) -> Dict[str, List[str]]:
        """
        Aggregates individual factual rules into a group consensus by finding
        the rules supported by a majority of members.
        """

        # Flatten the list of all rules from all users into a single list
        all_rules_flat = [
            rule
            for rules_list in individual_explanations.values()
            for rule in rules_list
        ]

        if not all_rules_flat:
            return {"unanimous": [], "majority": [], "minority": []}

        # Count the occurrences of each rule
        rule_counts = Counter(all_rules_flat)

        majority_threshold = (total_group_size // 2) + 1 if total_group_size > 1 else 1
        minority_threshold = 1
        cleaned_rules_set = self._clean_contradictory_rules(set(rule_counts.keys()))
        categorized_rules = {"unanimous": [], "majority": [], "minority": []}

        for rule in sorted(list(cleaned_rules_set)):
            count = rule_counts[rule]
            rule_with_support = f"{rule} ({count}/{total_group_size} members)"

            if count == total_group_size:
                categorized_rules["unanimous"].append(rule_with_support)
            elif count >= majority_threshold:
                categorized_rules["majority"].append(rule_with_support)
            elif count >= minority_threshold:
                categorized_rules["minority"].append(rule_with_support)

        return categorized_rules

    def _clean_contradictory_rules(self, rules_set: Set[str]) -> Set[str]:
        """Enhanced contradiction cleaning that handles genre rules"""
        conditions_by_attr = {}

        for rule in rules_set:
            # Handle genre rules
            if "Has genre:" in rule or "Does NOT have genre:" in rule:
                genre_match = re.search(r"`([^`]+)`", rule)
                if genre_match:
                    genre = genre_match.group(1)
                    attr = f"genre_{genre}"
                    op = "has" if "Has genre:" in rule else "not_has"
                    conditions_by_attr.setdefault(attr, set()).add(op)
            else:
                # Handle regular rules
                match = re.match(r"(.+?)\s*([<>]=?)\s*(\d+\.?\d*)", rule)
                if match:
                    attr, op, _ = match.groups()
                    conditions_by_attr.setdefault(attr.strip(), set()).add(op)

        # Find contradictory attributes
        invalid_attrs = set()
        for attr, ops in conditions_by_attr.items():
            if attr.startswith("genre_"):
                # Genre contradiction: has and not_has same genre
                if "has" in ops and "not_has" in ops:
                    invalid_attrs.add(attr)
            else:
                # Numerical contradiction: <= and >
                if any(op in ops for op in ["<=", "<"]) and any(
                    op in ops for op in [">", ">="]
                ):
                    invalid_attrs.add(attr)

        # Remove contradictory rules
        clean_rules = set()
        for rule in rules_set:
            is_invalid = False
            for invalid_attr in invalid_attrs:
                if invalid_attr.startswith("genre_"):
                    genre = invalid_attr.replace("genre_", "")
                    if f"`{genre}`" in rule:
                        is_invalid = True
                        break
                else:
                    if invalid_attr in rule:
                        is_invalid = True
                        break

            if not is_invalid:
                clean_rules.add(rule)

        return clean_rules

    def find_explanation(
        self,
        recommended_items: List[ItemId],
        members: List[UserId],
        user_hist: Dict[UserId, Set[ItemId]],
        dataset: pd.DataFrame,
        model=None,
        data_reader=None,
    ) -> Dict[str, Any]:
        """Enhanced explanation finding with tree storage for visualization"""
        if data_reader is None:
            raise ValueError(
                "A 'data_reader' object must be provided to find explanations."
            )

        detailed_explanations = {}
        explainable_count = 0

        if not recommended_items:
            return {"fidelity": 0.0, "details": {}}

        for item_id in recommended_items:
            all_individual_rules = {}
            all_counterfactuals = {}
            stored_classifiers = {}  # Store classifiers for visualization
            stored_metadata = {}  # Store metadata for visualization
            representative_decision_path = None
            threshold_info_for_item = None

            for user_id in members:
                user_id_consecutive = data_reader.get_new_user_id(user_id)
                clf, metadata = self._train_enhanced_decision_tree(
                    user_id_consecutive,
                    item_id,
                    user_hist.get(user_id, set()),
                    dataset,
                    model,
                    data_reader,
                )

                if clf and metadata:
                    if threshold_info_for_item is None and "threshold_info" in metadata:
                        threshold_info_for_item = metadata["threshold_info"]

                    explanation = self._generate_enhanced_individual_explanation(
                        clf, item_id, metadata
                    )

                    if explanation:
                        r, phi = explanation
                        all_individual_rules[user_id] = r
                        all_counterfactuals[user_id] = phi

                        if representative_decision_path is None:
                            representative_decision_path = r
                        # Store for visualization (use first successful classifier)
                        if not stored_classifiers:
                            stored_classifiers[user_id] = clf
                            stored_metadata[user_id] = metadata

            total_members_in_group = len(members)
            factual_set = self._aggregate_factual_rules(
                all_individual_rules, total_members_in_group
            )

            if representative_decision_path and factual_set:
                explainable_count += 1

                # Enhanced detailed explanations with visualization data
                item_explanation = {
                    "decision_path": representative_decision_path,
                    "group_factual_rule": factual_set,
                    "individual_counterfactuals": all_counterfactuals,
                }

                if threshold_info_for_item:
                    item_explanation["threshold_info"] = threshold_info_for_item

                # Add visualization data if available
                if stored_classifiers:
                    user_id_for_viz = list(stored_classifiers.keys())[0]
                    item_explanation.update(
                        {
                            "decision_tree": stored_classifiers[user_id_for_viz],
                            "feature_names": stored_metadata[user_id_for_viz].get(
                                "feature_labels", []
                            ),
                            "tree_metadata": stored_metadata[user_id_for_viz],
                            "item_genres": self.genre_profiles.get(str(item_id), set()),
                        }
                    )

                detailed_explanations[item_id] = item_explanation

        fidelity = (
            explainable_count / len(recommended_items) if recommended_items else 0.0
        )

        group_explanations = {
            "fidelity": fidelity,
            "details": detailed_explanations,
        }

        logging.info(
            f"Enhanced fidelity for {members}: {fidelity:.3f} ({explainable_count}/{len(recommended_items)})"
        )

        return group_explanations
