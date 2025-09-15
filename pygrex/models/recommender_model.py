from abc import ABC, abstractmethod
from typing import Dict, Union

from pygrex.data_reader.data_reader import DataReader


class RecommenderModel(ABC):
    """
    Abstract base class that defines the interface for recommendation models.
    All model implementations should inherit from this class.
    """

    @abstractmethod
    def predict(self, user_id: Union[str, int], item_id: Union[str, int]):
        """
        Make predictions for a specific user on a list of items.

        Args:
            user_id: The ID of the user
            item_ids: List of item IDs to predict ratings/scores for

        Returns:
            A dictionary mapping item IDs to predicted ratings/scores
        """
        pass

    @abstractmethod
    def fit(self, dataset: DataReader):
        """
        Train the model on data.
        The specific parameters depend on the model implementation.
        """
        pass
