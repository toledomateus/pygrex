import operator
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.signal import (
    find_peaks,
    peak_widths,
)

from pygrex.data_reader.data_reader import DataReader


class SlidingWindowEvaluator:
    """
    Evaluator of Stratigi's article for group recommendations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SlidingWindowEvaluator.

        Args:
            config: Configuration parameters for the evaluator
        """
        self.config = config
        self.group_predictions = None
        self.top_recommendation = None

    def set_group_recommender_values(
        self,
        group_predictions: Dict[Union[str, int], Dict[Union[str, int], float]],
        top_recommendation: Union[str, int],
    ) -> None:
        """
        Set group recommender values.

        Args:
            group_predictions: Dictionary mapping user IDs to their item predictions
            top_recommendation: List of top recommended items for the group
        """
        self.group_predictions = group_predictions
        self.top_recommendation = top_recommendation

    def evaluate(self, data: DataReader) -> Dict[str, Any]:
        """
        Evaluate the data using the Stratigis evaluator.

        Args:
            data: DataReader object containing dataset and transformation methods

        Returns:
            Dictionary with evaluation metrics
        """
        # Implementation would go here
        pass

    def calculate_item_popularity_score(
        self, items: List[Union[str, int]], data: DataReader
    ) -> Dict[Union[str, int], float]:
        """
        Calculate the normalized popularity of each item based on the number of interactions received.

        Args:
            items: List of item IDs
            data: Data object containing the dataset and transformation methods

        Returns:
            Dictionary with item IDs as keys and normalized popularity (0-1) as values
        """
        # Calculate popularity (number of interactions) for each item
        popularity_counts = {}
        for item_id in items:
            internal_item_id = data.get_new_item_id(item_id)
            count = len(data.dataset[data.dataset["itemId"] == internal_item_id])
            popularity_counts[item_id] = count

        # Find min and max values for normalization
        min_count = min(popularity_counts.values()) if popularity_counts else 0
        max_count = max(popularity_counts.values()) if popularity_counts else 0

        # Add 1% padding to the range
        range_value = max_count - min_count
        padded_range = range_value + (
            range_value / 50
        )  # Add 2% to range (1% on each end)
        padded_min = min_count - (
            range_value / 100
        )  # Subtract 1% of range from minimum

        # Normalize popularity values to [0,1]
        popularity_mask = {}
        for item_id, count in popularity_counts.items():
            popularity_mask[item_id] = (count - padded_min) / padded_range

        return popularity_mask

    def calculate_relevance_mask(
        self,
        target_item_id: Union[str, int],
    ) -> Dict[Union[str, int], float]:
        """
        Create a mapping between users and their prediction scores for a specific target item.

        Args:
            target_item_id :The ID of the item for which prediction scores are needed

        Returns:
            Dictionary mapping user IDs to their predicted scores for the target item
            Note: Users without a prediction for the target item will have a value of 0

        Examples
            >>> user_preds = {'user1': {'item1': 4.5, 'item2': 3.2}, 'user2': {'item2': 2.8}}
            >>> evaluator.set_group_recommender_values(user_preds,top_recommendation)
            >>> evaluator.calculate_relevance_mask('item1')
            {'user1': 4.5, 'user2': 0}
        """

        if self.group_predictions is None:
            raise ValueError(
                "User predictions not set. Call set_group_recommender_values first."
            )

        individual_predictions = {}

        for user_id, predictions in self.group_predictions.items():
            # Get the prediction for the target item if it exists, otherwise default to 0
            individual_predictions[user_id] = predictions.get(target_item_id, 0)

        return individual_predictions

    def calculate_relevance_score(
        self,
        item_id: Union[str, int],
        data: DataReader,
        prediction_scores: Dict[Union[str, int], float],
        members: List[Union[str, int]],
        rating_scale: tuple = (0, 5),  # Default rating scale
    ) -> float:
        """
        Calculate the normalized average prediction score for an item based on group members' predictions.

        Agrs
           item_id: ID of the item to calculate relevance for
            data : DataReader object containing dataset and ID mapping methods
            prediction_scores : Dictionary mapping user IDs to their prediction scores for items
            members : List of user IDs in the group
            rating_scale: Tuple indicating (min_rating, max_rating) for normalization

        Returns
            Normalized average prediction score in range [0,1]
            Returns 0 if no users in the group have interacted with the item

        Notes
            1. Calculates the average prediction score for the item from group members
            2. Normalizes the score to [0,1] range with 1% padding
        """
        total_score = 0
        valid_users_count = 0
        internal_item_id = data.get_new_item_id(item_id)

        for user_id in members:
            # Convert user ID to internal format
            internal_user_id = (
                data.get_new_user_id(int(user_id))
                if isinstance(user_id, (int, np.integer))
                else user_id
            )

            # Check if user has interacted with the item
            user_item_data = data.dataset[
                (data.dataset["userId"] == internal_user_id)
                & (data.dataset["itemId"] == internal_item_id)
            ]

            if user_item_data.empty:
                continue

            # Get the prediction score for this user
            if user_id in prediction_scores:
                total_score += prediction_scores[user_id]
                valid_users_count += 1

        # Return 0 if no valid users found
        if valid_users_count == 0:
            return 0

        # Calculate average score
        average_score = total_score / valid_users_count

        # Normalize to [0,1] with 1% padding
        min_value, max_value = rating_scale
        range_value = max_value - min_value
        padded_range = range_value + (
            range_value / 50
        )  # Add 2% to range (1% on each end)
        padded_min = min_value - (
            range_value / 100
        )  # Subtract 1% of range from minimum

        normalized_score = (average_score - padded_min) / padded_range
        return float(normalized_score)

    def calculate_item_intensity_score(
        self, item_id: Union[str, int], members: List[Union[str, int]], data: DataReader
    ) -> float:
        """
        Calculate what proportion of group members have interacted with the specified item.

        Args
            item_id : ID of the item to calculate interaction rate for
            members : List of user IDs in the group
            data : DataReader object containing dataset and ID mapping methods

        Returns
            Proportion of group members who have interacted with the item (range [0,1])
            0 means no group members have interacted with the item
            1 means all group members have interacted with the item
        """
        # Convert item ID to internal format
        internal_item_id = data.get_new_item_id(item_id)

        # Convert all user IDs to internal format
        internal_members = [data.get_new_user_id(user_id) for user_id in members]

        # Count how many users have interacted with the item
        interaction_count = len(
            data.dataset[
                (data.dataset.itemId == internal_item_id)
                & data.dataset.userId.isin(internal_members)
            ]
        )

        # Calculate proportion of group members who interacted with item
        if not members:
            return 0  # Avoid division by zero if no members

        interaction_rate = interaction_count / len(members)
        return interaction_rate

    def calculate_rating_score(
        self,
        item_id: Union[str, int],
        members: List[Union[str, int]],
        data: DataReader,
        rating_scale: tuple = (0, 5),
    ) -> float:
        """
        Calculate the normalized average rating given to an item by group members.

        Args
            item_id : ID of the item to calculate average rating for
            data : DataReader object containing dataset and ID mapping methods
            members : List of user IDs in the group
            rating_scale: Tuple indicating (min_rating, max_rating) for normalization

        Returns
            Normalized average rating in range [0,1]

        Notes
            - Considers all group members in the denominator even if some haven't rated the item
            - Normalizes the resulting average to [0,1] with 1% padding
        """
        # Convert item ID to internal format
        internal_item_id = data.get_new_item_id(item_id)

        # Convert all user IDs to internal format
        internal_members = [data.get_new_user_id(user_id) for user_id in members]

        # Get ratings from users who have rated this item
        rating_data = data.dataset[
            (data.dataset.itemId == internal_item_id)
            & data.dataset.userId.isin(internal_members)
        ]

        # Calculate average rating (sum of ratings divided by total group size)
        if len(members) == 0:
            return 0  # Avoid division by zero if no members

        total_rating = rating_data["rating"].sum()
        average_rating = total_rating / len(members)

        # Normalize to [0,1] with 1% padding
        min_value, max_value = rating_scale
        range_value = max_value - min_value
        padded_range = range_value + (
            range_value / 50
        )  # Add 2% to range (1% on each end)
        padded_min = min_value - (
            range_value / 100
        )  # Subtract 1% of range from minimum

        normalized_rating = (average_rating - padded_min) / padded_range
        return float(normalized_rating)

    def calculate_trending_score(
        self,
        members: List[Union[str, int]],
        item_id: Union[str, int],
        data: DataReader = None,
        peak_norm_min_height: float = 0.1,
        peak_norm_min_prominence: float = 0.05,
        peak_min_distance: int = 3,
        peak_width_rel_height: float = 0.5,
    ) -> tuple[float, Dict[Union[str, int], float], pd.DataFrame]:
        """
        Calculates a trending score for a user, using normalized data for hype period detection.

        Args
            members : List of user IDs in the group
            item_id : ID of the item to calculate trending score for
            data : DataReader object containing dataset and ID mapping methods
            peak_norm_min_height : Minimum height of peaks in normalized data to consider as significant
            peak_norm_min_prominence : Minimum prominence of peaks in normalized data
            peak_min_distance : Minimum distance between peaks in months
            peak_width_rel_height : Relative height for peak width calculation

        Returns
            tuple: (average_trending_score, individual_scores, hype_periods_for_item)
                average_trending_score: Average trending score across all group members (0-1)
                individual_scores: Dictionary mapping user IDs to their individual trending scores
                hype_periods_for_item: DataFrame containing detected hype periods for the item
        """

        if not members:
            print("Error: No group members provided for trending score calculation.")
            return 0.0, None, None

        _df = None
        if data is not None and isinstance(data, DataReader):
            _df = data.dataset.copy()
        else:
            # Fallback logic for loading _df
            if data is not None:
                print(
                    f"Warning: data was provided but is not a DataReader object (type: {type(type(data))})."
                )

        if _df.empty:
            print(
                "Error: The DataFrame (_df) is empty. Cannot calculate score or plot."
            )
            return 0.0, None, None

        required_columns = [
            "userId",
            "itemId",
            "rating",
            "timestamp",
        ]
        missing_columns = [col for col in required_columns if col not in _df.columns]
        if missing_columns:
            print(
                f"Error: Missing required columns in DataFrame: {', '.join(missing_columns)}"
            )
            return 0.0, None, None

        try:
            if "timestamp_dt" not in _df.columns or _df["timestamp_dt"].isnull().all():
                _df["timestamp_dt"] = pd.to_datetime(_df["timestamp"], unit="s")
            if "year_month" not in _df.columns or _df["year_month"].isnull().all():
                _df["year_month"] = _df["timestamp_dt"].dt.to_period("M")
        except Exception as e:
            print(f"Error during timestamp conversion or year-month extraction: {e}")
            return 0.0, None

        # Convert item ID to internal format
        internal_item_id = data.get_new_item_id(item_id)

        # Convert all user IDs to internal format
        internal_members = [data.get_new_user_id(user_id) for user_id in members]

        # Filter data for the specific item ID only
        item_df = _df[_df["itemId"] == internal_item_id]
        if item_df.empty:
            return 0.0, {user_id: 0.0 for user_id in members}, None

        # movie_ratings_per_month contains original rating counts
        movie_ratings_per_month = (
            item_df.groupby(["itemId", "year_month"], observed=False)
            .size()
            .reset_index(name="rating_count")
        )

        if movie_ratings_per_month.empty:
            return 0.0, {user_id: 0.0 for user_id in members}, None

        hype_periods_for_item = None

        # Process the specific item for hype period detection
        group_sorted = movie_ratings_per_month.sort_values("year_month").reset_index(
            drop=True
        )
        original_ratings = group_sorted["rating_count"].values

        # Normalization Step
        min_rating = original_ratings.min()
        max_rating = original_ratings.max()

        normalized_ratings = None
        if (
            max_rating > min_rating
        ):  # Avoid division by zero if all ratings are the same
            normalized_ratings = (original_ratings - min_rating) / (
                max_rating - min_rating
            )
        elif len(original_ratings) > 0:
            normalized_ratings = np.zeros_like(original_ratings, dtype=float)
        else:  # No ratings for this item in group_sorted (should not happen if groupby is correct)
            return 0.0, {user_id: 0.0 for user_id in members}, None

        # Peak Detection on Normalized Data
        peaks_indices, properties = find_peaks(
            normalized_ratings,
            height=peak_norm_min_height,
            distance=peak_min_distance,
            prominence=peak_norm_min_prominence,
        )

        hype_periods_list = []
        if len(peaks_indices) > 0:
            widths, _, left_ips, right_ips = peak_widths(
                normalized_ratings, peaks_indices, rel_height=peak_width_rel_height
            )

            for i, peak_idx in enumerate(peaks_indices):
                start_idx = max(0, int(round(left_ips[i])))
                end_idx = min(len(group_sorted) - 1, int(round(right_ips[i])))

                if start_idx <= end_idx:
                    start_month = group_sorted.iloc[start_idx]["year_month"]
                    end_month = group_sorted.iloc[end_idx]["year_month"]

                    hype_periods_list.append(
                        {
                            "itemId": item_id,
                            "hype_start_month": start_month,
                            "hype_end_month": end_month,
                            "peak_month": group_sorted.iloc[peak_idx]["year_month"],
                            "peak_rating_count_original": original_ratings[peak_idx],
                            "peak_rating_count_normalized": normalized_ratings[
                                peak_idx
                            ],
                        }
                    )

        if hype_periods_list:
            hype_periods_for_item = pd.DataFrame(hype_periods_list)
        else:
            print(
                f"No significant hype periods found for item {item_id} with current parameters (norm_min_height={peak_norm_min_height}, min_dist={peak_min_distance}, norm_min_prominence={peak_norm_min_prominence})."
            )
            return 0.0, {user_id: 0.0 for user_id in members}, pd.DataFrame()

        # Calculate trending scores for each user in the group
        individual_scores = {}
        valid_scores = []

        for idx, user_id in enumerate(internal_members):
            user_ratings = item_df[item_df["userId"] == user_id].copy()

            if user_ratings.empty:
                individual_scores[members[idx]] = 0.0
                continue

            # Merge user ratings with hype periods
            user_ratings_merged = pd.merge(
                user_ratings, hype_periods_for_item, on="itemId", how="left"
            )

            user_ratings_merged["is_match"] = (
                (
                    user_ratings_merged["year_month"]
                    >= user_ratings_merged["hype_start_month"]
                )
                & (
                    user_ratings_merged["year_month"]
                    <= user_ratings_merged["hype_end_month"]
                )
                & user_ratings_merged["hype_start_month"].notna()
            )

            if (
                not user_ratings_merged.empty
                and "is_match" in user_ratings_merged.columns
            ):
                is_event_trending = user_ratings_merged.groupby(
                    ["userId", "itemId", "timestamp_dt"]
                )["is_match"].any()
                num_trending_ratings = is_event_trending.sum()
                total_unique_rating_events = len(is_event_trending)
            else:
                num_trending_ratings = 0
                total_unique_rating_events = len(
                    user_ratings.drop_duplicates(
                        subset=["userId", "itemId", "timestamp_dt"]
                    )
                )

            if total_unique_rating_events == 0:
                individual_scores[members[idx]] = 0.0
            else:
                trending_score = num_trending_ratings / total_unique_rating_events
                individual_scores[members[idx]] = trending_score
                valid_scores.append(trending_score)

        # Calculate average trending score across all group members
        # Include users with 0.0 scores (no ratings for the item) in the average
        all_scores = [individual_scores[user_id] for user_id in members]
        average_trending_score = sum(all_scores) / len(members) if members else 0.0

        return average_trending_score, individual_scores, hype_periods_for_item

    def generate_ranked_items(
        self,
        all_rated_items: List[Union[str, int]],
        data: DataReader,
        group_members: List[Union[str, int]],
        component_weights: Dict[str, float] = None,
    ) -> List[Union[str, int]]:
        """
        Ranks items based on multiple scoring factors for a group of users.

        Calculates a composite score for each item based on:
        - Item popularity
        - Group preference intensity
        - Predicted ratings
        - Relevance to the group
        - Trends in the group

        Args:
            candidate_items: List of items that at least one group member has interacted with
            data: The DataReader object containing user-item interactions
            group_members: List of user identifiers in the group
            component_weights: Optional dictionary with weights for each component
                            (popularity, intensity, rating, relevance, trend)

        Returns:
            List of item IDs sorted in descending order by their composite scores
        """
        if self.group_predictions is None:
            raise ValueError(
                "User predictions not set. Call set_group_recommender_values first."
            )

        # Default weights if not provided
        if component_weights is None:
            component_weights = {
                "popularity": 1.0,
                "intensity": 1.0,
                "rating": 1.0,
                "relevance": 1.0,
                "trend": 1.0,
            }

        item_scores = {}
        popularity_scores = self.calculate_item_popularity_score(all_rated_items, data)

        relevance_mask = self.calculate_relevance_mask(self.top_recommendation)

        for item_id in all_rated_items:
            # Calculate individual score components

            popularity_score = popularity_scores[item_id]

            intensity_score = self.calculate_item_intensity_score(
                item_id, group_members, data
            )
            rating_score = self.calculate_rating_score(item_id, group_members, data)
            relevance_score = self.calculate_relevance_score(
                item_id, data, relevance_mask, group_members
            )

            trending_score, _, _ = self.calculate_trending_score(
                group_members,
                item_id,
                data,
                0.3,
                0.2,
                9,
                0.6,
            )

            composite_score = (
                component_weights["popularity"] * popularity_score
                + component_weights["intensity"] * intensity_score
                + component_weights["rating"] * rating_score
                + component_weights["relevance"] * relevance_score
                + component_weights["trend"] * trending_score
            )
            item_scores[item_id] = composite_score

        # Sort items by score in descending order
        ranked_items = sorted(
            item_scores.items(), key=operator.itemgetter(1), reverse=True
        )

        # Return just the sorted item IDs
        return [item_id for item_id, _ in ranked_items]
