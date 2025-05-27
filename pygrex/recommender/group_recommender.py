from typing import Dict, List, Union, Optional

import numpy as np

from pygrex.data_reader.data_reader import DataReader
from pygrex.models.recommender_model import RecommenderModel
from pygrex.utils.scale import Scale


class GroupRecommender:
    """
    A class to represent a group recommender system.
    """

    def __init__(self, data: DataReader):
        """Initialize the group recommender with data.

        Args:
            data: The dataset containing user-item interactions.
        """

        self.data = data
        self._group_predictions = None
        self._members = None
        self._item_pool = None
        self._model = None
        self._top_recommendation = None

    def setup_recommendation(
        self,
        model: RecommenderModel,
        members: List[Union[str, int]],
        data: DataReader,
    ) -> None:
        """
        Setup the group recommendation by specifying members, candidate items, and the model.

        Args:
            model: The recommendation model to use
            members: List of user IDs representing the group members
            data: DataReader object containing the dataset
        """
        self._members = members
        self._model = model

        # get all item IDs from the dataset
        item_ids = data.dataset["itemId"].unique()

        # Get items that no group member has interacted with
        self._item_pool = self.get_non_interacted_items_for_recommendation(
            self.data, item_ids, members
        )

        # Generate predictions for each group member
        self._group_predictions = self._generate_group_predictions()

    def _generate_group_predictions(self) -> Dict[Union[str, int], Dict[int, float]]:
        """
        Generate predictions for all group members.

        Returns:
            A dictionary with user IDs as keys and their predictions as values
        """
        if not self._members or self._model is None or self._item_pool is None:
            raise ValueError(
                "You must call setup_recommendation before generating predictions"
            )

        predictions = {}
        for member in self._members:
            user_pred = self.generate_recommendation(
                self._model, member, self._item_pool, self.data
            )
            predictions[member] = user_pred

        return predictions

    def get_non_interacted_items_for_recommendation(
        self,
        data: DataReader,
        item_ids: List[Union[str, int]],
        members: List[Union[str, int]],
    ) -> np.ndarray:
        """
        Returns the list of item IDs that none of the specified group members have interacted with.

        This method is typically used in recommendation systems to filter out items that have already
        been interacted with by any member of the group, ensuring that recommendations focus on new or
        unseen items.

        Args:
            data: The original dataset containing user-item interactions.
            item_ids: A list of all available item IDs to consider.
            members: A list of user IDs representing the group.

        Returns:
            np.ndarray: A list of item IDs that have not been interacted with by any member of the group.
        """
        # Get all unique item IDs interacted with by users in the group
        interacted_item_ids = data.dataset.loc[
            data.dataset.userId.isin(members), "itemId"
        ].unique()

        # Use numpy set difference to get non-interacted item IDs
        item_pool = np.setdiff1d(item_ids, interacted_item_ids, assume_unique=True)

        return item_pool

    def generate_recommendation(
        self,
        model: RecommenderModel,
        member: Union[str, int],
        item_pool: List[Union[str, int]],
        data: DataReader,
    ) -> Dict[int, float]:
        """
        Generate recommendations for a user based on the provided model.

        Args:
            model: A recommendation model that implements the RecommenderModel interface
            member: The ID of the user
            item_pool: List of item IDs to predict ratings/scores for
            data: The dataset containing user-item interactions

        Returns:
            A dictionary mapping item IDs to predicted ratings/scores
        """
        member = int(member)
        new_member_id = data.get_new_user_id(member)
        raw_predictions = []
        # Check if the model has item_factors and if the number of items matches the dataset
        for item in item_pool:
            item = int(item)  # Ensure item_id is treated as an intege
            raw_predictions.append(model.predict(new_member_id, item))

        # Ensure raw_predictions is a numpy array
        raw_predictions = np.array(raw_predictions)

        # Flatten the predictions if it's a 2D array (single user, multiple items)
        if raw_predictions.ndim == 2 and raw_predictions.shape[0] == 1:
            raw_predictions = raw_predictions.flatten()

        # Check if the length of raw_predictions matches item_pool
        if len(raw_predictions) != len(item_pool):
            raise ValueError(
                "Mismatch between predictions and item IDs. Check the model's predict function."
            )

        # Apply scaling with both methods
        scaled_linear = Scale.linear(
            raw_predictions,
            target_min=1,
            target_max=5,
        )
        # Convert the scaled predictions into a dictionary with item IDs as keys
        predictions = {
            item: scaled_pred for item, scaled_pred in zip(item_pool, scaled_linear)
        }

        # Sort the predictions in descending order of scores
        sorted_predictions = {}
        for item, score in predictions.items():
            # Ensure item_id is treated as an integer
            if isinstance(item, np.integer):
                item = int(item)
            item_original_id = data.get_original_item_id(item)
            # Since get_original_item_id returns a single value for integer input
            sorted_predictions[int(item_original_id)] = score

        # Sort the predictions in descending order of scores
        sorted_predictions = dict(
            sorted(sorted_predictions.items(), key=lambda item: item[1], reverse=True)
        )

        return sorted_predictions

    def get_group_recommendations(
        self, top_k: Optional[int] = None
    ) -> Union[int, List[int]]:
        """
        Generate recommendations for the group by aggregating individual predictions.

        Args:
            top_k: The number of recommendations to return.
                  If None, returns all recommendations sorted by score.
                  If 1, returns only the top recommendation as a single item ID.
                  If > 1, returns the top k recommendations as a list of item IDs.

        Returns:
            If top_k is 1, a single item ID. Otherwise, a list of item IDs.
        """
        if self._group_predictions is None:
            raise ValueError(
                "You must call setup_recommendation before getting recommendations"
            )

        sorted_scores = self.get_recommendation_scores()
        sorted_items = list(sorted_scores.items())

        # Return results based on top_k parameter
        if top_k is None:
            # Return all items as a list of (item_id, score) tuples
            return [item_id for item_id, _ in sorted_items]
        elif top_k == 1:
            # Return only the top item ID
            if sorted_items:
                return sorted_items[0][0]
            return None
        else:
            # Return top k item IDs
            return [
                item_id for item_id, _ in sorted_items[: min(top_k, len(sorted_items))]
            ]

    def get_top_recommendation(self) -> int:
        """
        Get the top recommendation for the group.

        Returns:
            The item ID with the highest average score across all group members.
        """
        if self._top_recommendation is None:
            self._top_recommendation = self.get_group_recommendations(top_k=1)
        return self._top_recommendation

    def get_recommendation_scores(self) -> Dict[int, float]:
        """
        Get the aggregated scores for all items across the group.

        Returns:
            A dictionary with item IDs as keys and their average scores as values.
        """
        if self._group_predictions is None:
            raise ValueError(
                "You must call setup_recommendation before getting recommendation scores"
            )

        # Aggregate scores for each item across all group members
        group_size = len(self._members)
        item_scores = {}

        for user, predictions in self._group_predictions.items():
            for item_id, score in predictions.items():
                if item_id in item_scores:
                    item_scores[item_id] += score
                else:
                    item_scores[item_id] = score

        # Calculate the average score for each item
        for item_id in item_scores:
            item_scores[item_id] /= group_size

        # Sort items by their scores in descending order
        sorted_scores = dict(
            sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_scores
