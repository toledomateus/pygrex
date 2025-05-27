import itertools
from typing import Dict, List, Union

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.data_reader.group_interaction_handler import GroupInteractionHandler
from pygrex.models.recommender_model import RecommenderModel
from pygrex.recommender.group_recommender import GroupRecommender


class SlidingWindowExplainer:
    """
    A class that uses a sliding window approach to find counterfactual explanations
    for group recommendation systems.

    This class helps identify which items, if removed from the group's interaction history,
    would cause a specific target item to no longer appear in the group recommendations.
    """

    def __init__(
        self,
        cfg: cfg,  # type: ignore
        data: DataReader,
        group_handler: GroupInteractionHandler,
        members: List[Union[str, int]],
        target_item: Union[str, int],
        candidate_items: List[Union[str, int]],
        sliding_window=None,
        model: RecommenderModel = None,
    ):
        """
        Initialize the SlidingWindowExplainer.

        Args:
            cfg: Configuration object with model parameters
            data: DataReader object containing the dataset
            group_handler: Object that handles group data modifications
            members: List of user IDs in the group
            target_item: The item ID for which explanation is sought
            candidate_items: List of candidate items for recommendations
            sliding_window: SlidingWindow object (if None, will need to be set later)
            model: Recommender model to use for predictions

        """
        self.data = data
        self.group_handler = group_handler
        self.members = members
        self.target_item = target_item
        self.candidate_items = candidate_items
        self.cfg = cfg
        self.sliding_window = sliding_window
        self.model = model

        # Results tracking
        self.explanations_found: Dict[int, List[Union[str, int]]] = {}
        self.calls = 0
        self.max_calls = 1000

    def set_sliding_window(self, sliding_window):
        """Set the sliding window object if not provided during initialization."""
        self.sliding_window = sliding_window

    def find_explanation(self) -> Dict[int, List[Union[str, int]]]:
        """
        Find counterfactual explanations using sliding window approach.

        Returns:
            dict: Dictionary of found explanations with call number as key
        """
        if not self.sliding_window:
            raise ValueError(
                "Sliding window has not been set. Use set_sliding_window() first."
            )

        # Get initial group recommendations to compare against
        original_group_rec = self.target_item
        if not original_group_rec:
            print("Could not find any items")
            return {}

        found = 0
        self.calls = 0
        wind_count = 0

        while True:
            # Get the sliding window
            big_window = self.sliding_window.get_next_window()

            # Check exit conditions
            if big_window is None or found > 0 or self.calls >= self.max_calls:
                break

            # Count calls and windows
            self.calls += 1
            wind_count += 1

            # Test if removing this window affects recommendations
            if self._test_window_removal(big_window, original_group_rec):
                # A counterfactual explanation has been found
                found += 1
                # Look for minimal subsets within this window
                self._find_minimal_subset(big_window, original_group_rec)

        if found == 0:
            print("Explanation could not be found")

        return self.explanations_found

    def _test_window_removal(
        self, item_ids: List[Union[str, int]], original_group_rec: Union[str, int]
    ) -> bool:
        """
        Test if removing the given items affects the group recommendation.

        Args:
            item_ids: List of item IDs to remove from group interactions
            original_group_rec: The original recommendation to compare against

        Returns:
            bool: True if removing these items changes recommendations, False otherwise
        """

        # Get new recommendations after removing items
        group_recommendation = self._get_recommendations_after_removal(item_ids)

        # Check if target item is still in recommendations

        return original_group_rec not in group_recommendation

    def _get_recommendations_after_removal(
        self, item_ids: List[Union[str, int]], top_n: int = 10
    ) -> List[Union[str, int]]:
        """
        Get group recommendations after removing specified items from interaction history.

        Args:
            item_ids: List of item IDs to remove from group interactions
            top_n: Number of top recommendations to return

        Returns:
            List of recommended item IDs
        """
        # Create modified dataset with items removed
        changed_data = self.group_handler.create_modified_dataset(
            original_data=self.data.dataset,
            group_ids=self.members,
            item_ids=item_ids,
            data=self.data,
        )

        # Create new DataReader and retrain model
        data_retrained = self._create_data_reader_and_prepare(changed_data)
        model_retrained = self._retrain_model(data_retrained)

        # Set up recommender with new model and data
        group_recommender = GroupRecommender(data_retrained)
        group_recommender.setup_recommendation(
            model_retrained, self.members, data_retrained
        )

        # Return new recommendations
        return group_recommender.get_group_recommendations(top_n)

    def _create_data_reader_and_prepare(self, changed_data):
        """
        Create and prepare a new DataReader with modified data.

        Args:
            changed_data: DataFrame with modified dataset

        Returns:
            DataReader: A new DataReader object with the modified dataset
        """
        data_retrained = DataReader(
            filepath_or_buffer=None,
            sep=None,
            names=None,
            skiprows=0,
            dataframe=changed_data,
        )

        # Fix for potential dataset issue in original code
        # data_retrained.dataset = data_retrained.dataset.iloc[1:].reset_index(drop=True)

        # Prepare data
        data_retrained.make_consecutive_ids_in_dataset()
        data_retrained.binarize(binary_threshold=1)

        return data_retrained

    def _retrain_model(self, data):
        """
        Retrain the recommendation model with modified data.

        Args:
            data: Prepared DataReader object with modified dataset

        Returns:
            Retrained model
        """
        self.model.fit(data)
        return self.model

    def _find_minimal_subset(
        self, big_window: List[Union[str, int]], original_group_rec: Union[str, int]
    ) -> None:
        """
        Find minimal subset of items that act as counterfactual explanation.

        Args:
            big_window: List of item IDs to search within
            original_group_rec: The original recommendation to compare against

        """
        found_subset = 0

        # Try combinations of different lengths
        for length in range(1, len(big_window) + 1):
            if found_subset > 0 or self.calls > self.max_calls:
                break

            combinations = itertools.combinations(big_window, length)
            for item_combo in combinations:
                if found_subset > 0 or self.calls > self.max_calls:
                    break

                subset_items = list(item_combo)
                self.calls += 1

                # Get recommendations after removing this subset
                new_recommendations = self._get_recommendations_after_removal(
                    subset_items
                )

                # Check if this is a counterfactual explanation
                if original_group_rec not in new_recommendations:
                    found_subset += 1
                    self._record_explanation(
                        subset_items, original_group_rec, new_recommendations[0]
                    )

    def _record_explanation(
        self,
        explanation_items: List[Union[str, int]],
        original_rec: Union[str, int],
        new_rec: Union[str, int],
    ) -> None:
        """
        Record and display found explanation.

        Args:
            explanation_items: Items that form the counterfactual explanation
            original_rec: Original recommendation
            new_rec: New top recommendation after removing explanation items
        """
        print(
            f"If the group had not interacted with these items {explanation_items},\n"
            f"the item of interest {original_rec} would not have appeared on the recommendation list;\n"
            f"instead, {new_rec} would have been recommended."
        )
        print("")
        print(f"Explanation: {explanation_items} : found at call: {self.calls}")

        # Calculate metrics for the explanation
        item_intensity = self._calculate_item_intensity(explanation_items)
        user_intensity = self._calculate_user_intensity(explanation_items)

        self.explanations_found[self.calls] = explanation_items
        exp_size = len(explanation_items)

        print(f"{exp_size}\t{self.calls}\t{item_intensity}\t{user_intensity}")

    def _calculate_item_intensity(self, items: List[Union[str, int]]) -> List[float]:
        """
        Calculate average item intensity for explanation items.

        Args:
            items: List of item IDs in the explanation

        Returns:
            List of average intensity scores for each item
        """

        return self._calculate_average_item_intensity_score(
            items, self.members, self.data
        )

    def _calculate_user_intensity(self, items: List[Union[str, int]]) -> List[float]:
        """
        Calculate user intensity score for explanation items.

        Args:
            items: List of item IDs in the explanation

        Returns:
            List of intensity scores for each user
        """
        return self._calculate_user_intensity_score(items, self.members, self.data)

    @staticmethod
    def _calculate_average_item_intensity_score(
        explanation: List[Union[str, int]],
        members: List[Union[str, int]],
        data: DataReader,
    ) -> List[float]:
        """
        Calculate the average item intensity for a counterfactual explanation.

        Average item intensity is defined as the average number of interactions
        between group members and each item in the explanation.

        Args:
            explanation: The counterfactual explanation items.
            members: User IDs of the group members.
            data: DataReader object containing the dataset and ID mapping methods.

        Returns:
            list: Average intensity for each item in the explanation.
        """
        # Convert user IDs to internal representation
        internal_group_ids = [int(data.get_new_user_id(user_id)) for user_id in members]

        group_size = len(members)
        item_intensities = []

        for item_id in explanation:
            # Convert item ID to internal representation
            internal_item_id = data.get_new_item_id(item_id)

            # Count interactions between this item and group members
            interactions_count = len(
                data.dataset[
                    (data.dataset.itemId == internal_item_id)
                    & (data.dataset.userId.isin(internal_group_ids))
                ]
            )

            # Calculate average intensity
            average_intensity = interactions_count / group_size
            item_intensities.append(average_intensity)

        return item_intensities

    @staticmethod
    def _calculate_user_intensity_score(
        explanation_items: List[Union[str, int]],
        members: List[Union[str, int]],
        data: DataReader,
    ) -> List[float]:
        """
        Calculate the interaction intensity for each user based on their interactions with items in an explanation.

        Interaction intensity represents how much a user has interacted with the items in the explanation,
        normalized by the total number of explanation items.

        Args
            explanation_items : List of item IDs in the explanation
            members : List of user IDs to calculate intensity for
            data : DataReader object containing the dataset and ID mapping methods

        Returns
            List of interaction intensities for each user (same order as members)
            Values range from 0 to 1, where:
            - 0 means no interaction with any explanation item
            - 1 means interaction with all explanation items

        Notes
            Intensity is calculated as: (number of user interactions with explanation items) / (number of explanation items)
        """
        # Convert external item IDs to internal IDs
        internal_item_ids = [
            data.get_new_item_id(item_id) for item_id in explanation_items
        ]

        user_intensities = []
        num_explanation_items = len(explanation_items)

        for member in members:
            # Convert external user ID to internal ID
            internal_user_id = data.get_new_user_id(member)

            # Count interactions between this user and explanation items
            user_interactions_count = len(
                data.dataset[
                    (data.dataset.itemId.isin(internal_item_ids))
                    & (data.dataset.userId == internal_user_id)
                ]
            )

            # Calculate intensity as proportion of explanation items the user interacted with
            intensity = user_interactions_count / num_explanation_items
            user_intensities.append(intensity)

        return user_intensities
