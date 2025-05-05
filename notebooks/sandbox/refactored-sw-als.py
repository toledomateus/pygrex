# Standard library imports
import itertools
import math # Added for potential percentage window calculation
from operator import itemgetter
import sys # Added for error exit
from typing import List, Dict, Any, Optional, Union, Tuple, Set

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import stats

# Local application/library specific imports
# --- ASSUMPTION: These classes exist and work as expected ---
# from recoxplainer.config import cfg # Assuming cfg is loaded elsewhere
# from recoxplainer.data_reader import DataReader
# from recoxplainer.models.als_model import ALS
# --- Placeholder classes for demonstration ---
class CfgObject: # Placeholder
    def __init__(self):
        self.data = type('obj', (object,), {'test': {'filepath_or_buffer': 'dummy_path'}})()
        self.model = type('obj', (object,), {'als': {'factors': 10}})()
cfg = CfgObject() # Placeholder instance

class DataReader: # Placeholder definition
    def __init__(self, filepath_or_buffer=None, sep=None, names=None, groups_filepath=None, skiprows=0, dataframe=None, **kwargs):
        if dataframe is not None:
            self.dataset = dataframe.copy() # Work with a copy
        else:
            # In a real scenario, load from filepath_or_buffer
            print(f"Warning: Placeholder DataReader initialized without real data loading from {filepath_or_buffer}")
            self.dataset = pd.DataFrame({'userId': [0, 0, 1, 1, 2], 'itemId': [0, 1, 1, 2, 0], 'rating': [4, 3, 5, 2, 5]}) # Dummy data
        self.user_map_internal_to_original = {i: i for i in self.dataset['userId'].unique()}
        self.user_map_original_to_internal = {v: k for k, v in self.user_map_internal_to_original.items()}
        self.item_map_internal_to_original = {i: i for i in self.dataset['itemId'].unique()}
        self.item_map_original_to_internal = {v: k for k, v in self.item_map_internal_to_original.items()}
        self.num_users = self.dataset['userId'].nunique()
        self.num_items = self.dataset['itemId'].nunique()

    def make_consecutive_ids_in_dataset(self):
        # Placeholder: Assume IDs are already consecutive or mapping handles it
        pass

    def binarize(self, binary_threshold=1):
        # Placeholder
        if 'rating' in self.dataset.columns:
             self.dataset['rating'] = (self.dataset['rating'] >= binary_threshold).astype(int)

    def get_new_user_id(self, original_user_id: Union[int, np.integer, str]) -> Optional[int]:
        """Gets the internal user ID for an original user ID."""
        original_user_id = int(original_user_id) if isinstance(original_user_id, (int, np.integer)) else original_user_id
        return self.user_map_original_to_internal.get(original_user_id)

    def get_new_item_id(self, original_item_id: Union[int, np.integer, str]) -> Optional[int]:
        """Gets the internal item ID for an original item ID."""
        original_item_id = int(original_item_id) if isinstance(original_item_id, (int, np.integer)) else original_item_id
        return self.item_map_original_to_internal.get(original_item_id)

    def get_original_user_id(self, internal_user_id: int) -> Optional[Union[int, str]]:
         """Gets the original user ID from an internal ID."""
         return self.user_map_internal_to_original.get(internal_user_id)

    def get_original_item_id(self, internal_item_id: Union[int, np.ndarray]) -> Optional[Union[int, str, List[Union[int, str]]]]:
        """Gets the original item ID(s) from internal ID(s)."""
        if isinstance(internal_item_id, (int, np.integer)):
            return self.item_map_internal_to_original.get(int(internal_item_id))
        elif isinstance(internal_item_id, np.ndarray):
            return [self.item_map_internal_to_original.get(int(i)) for i in internal_item_id if int(i) in self.item_map_internal_to_original]
        return None # Or raise error

    def get_new_user_ids(self, user_ids: List[Union[int, np.integer, str]]) -> List[int]:
        """Converts a list of original user IDs to internal IDs."""
        return [self.get_new_user_id(uid) for uid in user_ids if self.get_new_user_id(uid) is not None]

    def get_new_item_ids(self, item_ids: List[Union[int, np.integer, str]]) -> List[int]:
        """Converts a list of original item IDs to internal IDs."""
        return [self.get_new_item_id(iid) for iid in item_ids if self.get_new_item_id(iid) is not None]

    def get_original_item_ids(self, internal_item_ids: List[int]) -> List[Union[int, str]]:
         """Gets the original item IDs from a list of internal IDs."""
         return [self.get_original_item_id(iid) for iid in internal_item_ids if self.get_original_item_id(iid) is not None]

    def remove_interactions(self, user_ids: List[Union[int, str]], item_ids: List[Union[int, str]]) -> pd.DataFrame:
        """Returns a *new* DataFrame with specified original user-item interactions removed."""
        internal_user_ids = self.get_new_user_ids(user_ids)
        internal_item_ids = self.get_new_item_ids(item_ids)

        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        if not internal_user_ids or not internal_item_ids:
             print("Warning: No valid internal IDs found for users or items to remove.")
             return self.dataset.copy() # Return copy if no valid IDs

        # Create a boolean mask for rows to drop
        mask_to_drop = (
            self.dataset['userId'].isin(internal_user_ids) &
            self.dataset['itemId'].isin(internal_item_ids)
        )

        # Return a *new* DataFrame with the rows dropped
        return self.dataset.drop(self.dataset[mask_to_drop].index).copy()


class ALS: # Placeholder definition
    def __init__(self, **kwargs):
        print(f"Placeholder ALS initialized with args: {kwargs}")
        self.user_factors = None
        self.item_factors = None
        self.user_map = None
        self.item_map = None

    def fit(self, data_reader: DataReader):
        print("Placeholder ALS: Fitting model...")
        # Simulate factor creation based on the data_reader's dataset
        self.num_users = data_reader.num_users
        self.num_items = data_reader.num_items
        self.user_map = data_reader.user_map_original_to_internal # Store mapping used during training
        self.item_map = data_reader.item_map_original_to_internal # Store mapping used during training
        # Example: Random factors (replace with actual ALS logic)
        factors = 10 # Example factor count
        self.user_factors = np.random.rand(self.num_users, factors)
        self.item_factors = np.random.rand(self.num_items, factors)
        print(f"Placeholder ALS: Fit complete. Users={self.num_users}, Items={self.num_items}")


    def predict(self, internal_user_id: int, internal_item_id: int) -> Optional[float]:
        """Predicts rating for a single internal user-item pair."""
        if self.user_factors is None or self.item_factors is None:
            print("Error: Model not fitted.")
            return None
        if internal_user_id >= self.num_users or internal_item_id >= self.num_items or internal_user_id < 0 or internal_item_id < 0:
             # print(f"Warning: Predict called with invalid internal IDs: User {internal_user_id} (max {self.num_users-1}), Item {internal_item_id} (max {self.num_items-1})")
             # This can happen if the item/user wasn't in the training data for *this specific retraining*
             return 0.0 # Return a default low score for items not trainable

        # Simple dot product prediction (replace with actual ALS prediction)
        # print(f"Predicting for internal user {internal_user_id}, internal item {internal_item_id}")
        # print(f"User factor shape: {self.user_factors.shape}, Item factor shape: {self.item_factors.shape}")
        prediction = np.dot(self.user_factors[internal_user_id, :], self.item_factors[internal_item_id, :])
        return float(prediction)
# --- End of Placeholder classes ---


# --- Configuration Constants ---
# Paths
BASE_DATASET_PATH = "../datasets/ml-100k/" # Adjust as needed
GROUPS_FILE_PATH = f"{BASE_DATASET_PATH}groupsWithHighRatings5.txt" # Example path

# Model & Data Parameters (Examples, adjust based on cfg or actual needs)
ALS_FACTORS = cfg.model.als.get('factors', 50) # Example: get from cfg or default
ALS_REGULARIZATION = cfg.model.als.get('regularization', 0.01)
ALS_ITERATIONS = cfg.model.als.get('iterations', 15)
BINARY_THRESHOLD = 1.0

# Explanation Parameters
DEFAULT_GROUP_SIZE_FOR_AVG = 5 # Used in groupRecommendations if size isn't derived
RECOMMENDATION_LIST_SIZE = 10 # Top-N recommendations to check against (flag > 0)
WINDOW_SIZE = 3 # Static window size (can be changed to percentage)
# WINDOW_PERCENTAGE = 0.1 # Example if using percentage: math.floor(len(chart) * WINDOW_PERCENTAGE)
MAX_RECOMMENDER_CALLS = 1000 # Safety break for explanation search

# Scaling Parameters
TARGET_RATING_MIN = 1.0
TARGET_RATING_MAX = 5.0
DEFAULT_RAW_PREDICTION_MIN = 0.0 # Default reference min for scaling
DEFAULT_RAW_PREDICTION_MAX = 6.0 # Default reference max for scaling (adjust if ALS outputs different range)

# --- Helper Classes ---

class SlidingWindow:
    """Class for initiating and keeping track of the sliding window."""

    def __init__(self, items: List[Any], window_size: int):
        """Initiate the class.

        Args:
            items (List[Any]): The list of items to slide over.
            window_size (int): The size of the window. Must be > 0.
        """
        if window_size <= 0:
             raise ValueError("Window size must be positive.")
        self.items = items
        self.window_size = window_size
        self.index = 0  # Keep track of the start of the current window

    def get_next_window(self) -> Optional[List[Any]]:
        """Returns the next window and advances the index.

        Returns:
            Optional[List[Any]]: The next window list, or None if all windows are processed.
        """
        start_index = self.index
        end_index = start_index + self.window_size
        if end_index <= len(self.items):
            window = self.items[start_index:end_index]
            self.index += 1  # Move starting point for the *next* window
            return window
        else:
            # No more full windows available
            return None


# --- Core Functions ---

def change_data(data_reader: DataReader, group_ids: List[Union[int, str]],
                item_ids: List[Union[int, str]]) -> pd.DataFrame:
    """
    Creates a *new* DataFrame by removing interactions of specific group members
    with specific items from the original data.

    Args:
        data_reader (DataReader): The data reader instance holding the original dataset and mappings.
        group_ids (List[Union[int, str]]): List of original IDs of the group members.
        item_ids (List[Union[int, str]]): List of original item IDs (potential counterfactual).

    Returns:
        pd.DataFrame: A new DataFrame with the specified interactions removed.
    """
    if not isinstance(data_reader, DataReader):
         raise TypeError("`data_reader` must be an instance of DataReader.")
    # Use the refactored method in DataReader
    return data_reader.remove_interactions(group_ids, item_ids)


def read_groups(file_path: str) -> List[str]:
    """
    Reads group IDs (as strings) from a file, one group per line.

    Args:
        file_path (str): The path to the file containing group IDs.

    Returns:
        List[str]: A list of group ID strings (e.g., "user1_user2_user3").

    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    try:
        with open(file_path, "r") as f:
            # Read lines and strip any leading/trailing whitespace (like newlines)
            groups = [line.strip() for line in f if line.strip()]
        return groups
    except FileNotFoundError:
        print(f"Error: Group file not found at {file_path}")
        raise # Re-raise the exception


def get_group_members(group_string: str) -> List[int]:
    """
    Parses a group ID string (e.g., "12_34_56") into a list of integer member IDs.

    Args:
        group_string (str): The group ID string, typically '_' delimited.

    Returns:
        List[int]: A list of the group members' integer IDs.
    """
    group_string = group_string.strip()
    members = group_string.split("_")
    member_ids = [int(m) for m in members if m.isdigit()] # Ensure conversion to int
    return member_ids


def get_rated_items_by_group_members(group_member_ids: List[int], data_reader: DataReader) -> List[Union[int, str]]:
    """
    Returns a list of original item IDs that at least one group member has interacted with.

    Args:
        group_member_ids (List[int]): A list of the original group member IDs.
        data_reader (DataReader): The data reader instance.

    Returns:
        List[Union[int, str]]: A list of unique original item IDs rated by the group.
    """
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)
    if not internal_group_ids:
        return [] # No valid members found in data

    # Filter dataset for interactions by these internal user IDs
    rated_internal_items = data_reader.dataset[
        data_reader.dataset['userId'].isin(internal_group_ids)
    ]['itemId'].unique()

    # Convert internal item IDs back to original IDs
    original_item_ids = data_reader.get_original_item_ids(list(rated_internal_items)) # Use list conversion

    return original_item_ids


def get_items_for_recommendation(
    all_item_ids: np.ndarray, # Should be internal IDs for efficiency
    group_member_ids: List[int],
    data_reader: DataReader
) -> np.ndarray:
    """
    Returns internal item IDs that *no one* in the group has interacted with.

    Args:
        all_item_ids (np.ndarray): A NumPy array of *all unique internal* item IDs in the dataset.
        group_member_ids (List[int]): A list of the original group member IDs.
        data_reader (DataReader): The data reader instance.

    Returns:
        np.ndarray: A NumPy array of internal item IDs not interacted with by the group.
    """
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)
    if not internal_group_ids:
        return all_item_ids # If group is empty or not in data, all items are candidates

    # Get internal item IDs interacted with by the group
    items_rated_by_group = data_reader.dataset.loc[
        data_reader.dataset['userId'].isin(internal_group_ids), 'itemId'
    ].unique()

    # Find the difference: items in all_item_ids that are NOT in items_rated_by_group
    # Ensure both arrays contain internal IDs of the same type (e.g., int)
    items_to_predict_for = np.setdiff1d(all_item_ids, items_rated_by_group, assume_unique=True)

    return items_to_predict_for


def scale_predictions(
    raw_predictions: np.ndarray,
    target_min: float = TARGET_RATING_MIN,
    target_max: float = TARGET_RATING_MAX,
    method: str = "linear",
    # Removed ref_min/ref_max as default, calculate from data or use explicit bounds if needed
) -> np.ndarray:
    """
    Scales raw predictions to the target range [target_min, target_max].

    Args:
        raw_predictions (np.ndarray): The raw prediction values.
        target_min (float): Minimum of the target range.
        target_max (float): Maximum of the target range.
        method (str): Scaling method ('linear' or 'quantile').

    Returns:
        np.ndarray: Scaled predictions.

    Raises:
        ValueError: If raw_predictions is empty or method is invalid.
    """
    if raw_predictions.size == 0:
        # Return empty array if input is empty, instead of raising error immediately
        # This might happen if prediction fails for all candidate items.
        print("Warning: scale_predictions received empty array.")
        return np.array([])

    raw_predictions = np.array(raw_predictions) # Ensure it's a numpy array

    if method == "linear":
        # Option 1: Simple min-max scaling on the observed raw predictions
        min_raw = np.min(raw_predictions)
        max_raw = np.max(raw_predictions)

        if max_raw == min_raw:
            # If all predictions are the same, map to the middle of the target range
            scaled_predictions = np.full_like(raw_predictions, (target_min + target_max) / 2)
        else:
            scaled_predictions = target_min + (raw_predictions - min_raw) * (
                target_max - target_min
            ) / (max_raw - min_raw)

        # Option 2 (Closer to original but potentially complex): Use clipping + reference bounds
        # q1, q3 = np.percentile(raw_predictions, [25, 75])
        # iqr = q3 - q1
        # lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr
        # clipped_predictions = np.clip(raw_predictions, lower_bound, upper_bound)
        # min_clipped = np.min(clipped_predictions)
        # max_clipped = np.max(clipped_predictions)
        # if max_clipped == min_clipped:
        #     # Handle constant case after clipping (e.g., map to midpoint or use a fixed reference)
        #     scaled_predictions = np.full_like(raw_predictions, (target_min + target_max) / 2)
        # else:
        #     # Scale based on the clipped range
        #     scaled_predictions = target_min + (np.clip(raw_predictions, min_clipped, max_clipped) - min_clipped) * (
        #         target_max - target_min
        #     ) / (max_clipped - min_clipped)

    elif method == "quantile":
        # Quantile-based scaling maps ranks to the target range
        if len(raw_predictions) == 1:
             # Handle single prediction case for quantile (map to midpoint)
             scaled_predictions = np.full_like(raw_predictions, (target_min + target_max) / 2)
        else:
            ranks = stats.rankdata(raw_predictions, method="average")
            # Scale ranks to [0, 1] then to [target_min, target_max]
            scaled_predictions = target_min + (ranks - 1) * (target_max - target_min) / (
                len(raw_predictions) - 1
            )
    else:
        raise ValueError("Invalid scaling method. Choose 'linear' or 'quantile'.")

    # Ensure final scaled predictions are strictly within [target_min, target_max]
    scaled_predictions = np.clip(scaled_predictions, target_min, target_max)

    return scaled_predictions


def generate_recommendations_for_user(
    model: ALS,
    user_id_original: Union[int, str],
    item_ids_to_predict_internal: np.ndarray, # Use internal IDs for prediction
    data_reader: DataReader
) -> Dict[Union[int, str], float]:
    """
    Generates scaled predictions for a single user for a list of candidate item IDs.

    Args:
        model (ALS): The trained recommendation model.
        user_id_original (Union[int, str]): The original ID of the user.
        item_ids_to_predict_internal (np.ndarray): *Internal* IDs of items to get predictions for.
        data_reader (DataReader): The data reader instance used for training this model.

    Returns:
        Dict[Union[int, str], float]: Dictionary mapping original item IDs to scaled prediction scores,
                                      sorted descending by score. Returns empty dict if user is invalid.
    """
    internal_user_id = data_reader.get_new_user_id(user_id_original)

    # Ensure the user was known to the model (i.e., was in the training data used for *this* model instance)
    # Note: We check against the model's internal map if available, or the data_reader's map otherwise.
    # The check `internal_user_id >= model.num_users` in `model.predict` should also catch this.
    if internal_user_id is None:
        print(f"Warning: User {user_id_original} not found in data reader mapping for prediction.")
        return {}

    raw_predictions = []
    valid_internal_item_ids = []

    for internal_item_id in item_ids_to_predict_internal:
        # Ensure item_id is integer
        internal_item_id = int(internal_item_id)
        pred = model.predict(internal_user_id, internal_item_id)
        # pred can be None if model predict fails, or 0.0 for unknown items
        if pred is not None:
            raw_predictions.append(pred)
            valid_internal_item_ids.append(internal_item_id)
        # else: handle prediction failure? Maybe log it.

    if not raw_predictions:
        # No valid predictions could be made (e.g., all candidate items were unknown to the model)
        return {}

    raw_predictions_np = np.array(raw_predictions)
    scaled_predictions = scale_predictions(raw_predictions_np, method="linear") # Or "quantile"

    # Create dict mapping ORIGINAL item IDs to scaled scores
    predictions_dict = {}
    original_item_ids = data_reader.get_original_item_ids(valid_internal_item_ids)

    if len(original_item_ids) != len(scaled_predictions):
         print(f"Warning: Mismatch between original item IDs ({len(original_item_ids)}) and scaled predictions ({len(scaled_predictions)}) for user {user_id_original}.")
         # Attempt to pair based on index, but this indicates a potential issue in ID mapping or filtering.
         min_len = min(len(original_item_ids), len(scaled_predictions))
         for i in range(min_len):
              predictions_dict[original_item_ids[i]] = scaled_predictions[i]
    else:
        for original_id, score in zip(original_item_ids, scaled_predictions):
            predictions_dict[original_id] = score


    # Sort the dictionary by scores in descending order
    sorted_predictions = dict(
        sorted(predictions_dict.items(), key=itemgetter(1), reverse=True)
    )

    return sorted_predictions


def get_group_recommendations(
    member_predictions: Dict[Union[int, str], Dict[Union[int, str], float]],
    aggregation_strategy: str = "average", # Example: add more strategies later
    top_n: Optional[int] = None
) -> Union[List[Union[int, str]], Union[int, str]]:
    """
    Aggregates individual member predictions into a group recommendation list.

    Args:
        member_predictions (Dict[Union[int, str], Dict[Union[int, str], float]]):
            Nested dictionary: {member_id: {original_item_id: score, ...}, ...}.
        aggregation_strategy (str): How to combine scores (currently only "average").
        top_n (Optional[int]):
            - If None, return the full ranked list of items.
            - If 1, return only the single top item ID.
            - If > 1, return a list of the top N item IDs.

    Returns:
        Union[List[Union[int, str]], Union[int, str]]: The group recommendation(s).
            Returns an empty list or None if no predictions available.
    """
    if not member_predictions:
        return [] if top_n != 1 else None

    group_scores: Dict[Union[int, str], float] = {}
    item_counts: Dict[Union[int, str], int] = {} # For averaging

    # Collect scores for each item across all members
    for member_id, predictions in member_predictions.items():
        for item_id, score in predictions.items():
            group_scores[item_id] = group_scores.get(item_id, 0) + score
            item_counts[item_id] = item_counts.get(item_id, 0) + 1

    if not group_scores:
         return [] if top_n != 1 else None

    # Calculate final group score based on strategy
    final_group_predictions: Dict[Union[int, str], float] = {}
    if aggregation_strategy == "average":
        num_members = len(member_predictions) # Use actual number of members with predictions
        if num_members == 0: return [] if top_n != 1 else None

        # Average score per item (alternative: divide by item_counts[item_id] if members rated different subsets)
        for item_id, total_score in group_scores.items():
             # Decide how to handle items predicted for only a subset of members.
             # Option 1: Average over all group members (implicit score of 0 for non-predicted?)
             # final_group_predictions[item_id] = total_score / num_members
             # Option 2: Average over only members for whom prediction exists
             final_group_predictions[item_id] = total_score / item_counts[item_id]

    else:
        raise NotImplementedError(f"Aggregation strategy '{aggregation_strategy}' not implemented.")

    # Sort the final group predictions by score
    sorted_group_predictions = dict(
        sorted(final_group_predictions.items(), key=itemgetter(1), reverse=True)
    )

    # Return based on top_n
    sorted_items = list(sorted_group_predictions.keys())

    if top_n == 1:
        return sorted_items[0] if sorted_items else None
    elif top_n is not None and top_n > 1:
        return sorted_items[:top_n]
    else: # top_n is None
        return sorted_items


# --- Explanation Quality Metrics ---

def calculate_average_item_intensity(
    explanation_item_ids: List[Union[int, str]],
    group_member_ids: List[int],
    data_reader: DataReader
) -> List[float]:
    """
    Calculates the intensity for each item in the explanation set.
    Item intensity = (Number of group members who interacted with the item) / (Total group size).

    Args:
        explanation_item_ids (List[Union[int, str]]): Original IDs of items in the explanation.
        group_member_ids (List[int]): Original IDs of group members.
        data_reader (DataReader): Data reader instance.

    Returns:
        List[float]: List of average item intensities, one per item in the explanation.
                     Returns empty list if group or explanation is empty or invalid.
    """
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)
    internal_expl_item_ids = data_reader.get_new_item_ids(explanation_item_ids)
    group_size = len(group_member_ids) # Use original list size for denominator

    if not internal_group_ids or not internal_expl_item_ids or group_size == 0:
        return []

    intensities = []
    interactions = data_reader.dataset # Use the base dataset for calculation

    for internal_item_id in internal_expl_item_ids:
        # Count how many members in the group interacted with this specific item
        num_interactions = interactions[
            (interactions['itemId'] == internal_item_id) &
            (interactions['userId'].isin(internal_group_ids))
        ]['userId'].nunique() # Count distinct users who interacted

        intensity = num_interactions / group_size if group_size > 0 else 0.0
        intensities.append(intensity)

    return intensities


def calculate_user_intensity(
    explanation_item_ids: List[Union[int, str]],
    group_member_ids: List[int],
    data_reader: DataReader
) -> List[float]:
    """
    Calculates the intensity for each user in the group based on the explanation set.
    User intensity = (Number of items in explanation the user interacted with) / (Total explanation size).

    Args:
        explanation_item_ids (List[Union[int, str]]): Original IDs of items in the explanation.
        group_member_ids (List[int]): Original IDs of group members.
        data_reader (DataReader): Data reader instance.

    Returns:
        List[float]: List of user intensities, one per user in the group.
                     Returns empty list if group or explanation is empty or invalid.
    """
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)
    internal_expl_item_ids = data_reader.get_new_item_ids(explanation_item_ids)
    explanation_size = len(explanation_item_ids) # Use original list size for denominator

    if not internal_group_ids or not internal_expl_item_ids or explanation_size == 0:
        return []

    intensities = []
    interactions = data_reader.dataset

    for internal_user_id in internal_group_ids:
        # Count how many items in the explanation this specific user interacted with
        num_interactions = interactions[
            (interactions['userId'] == internal_user_id) &
            (interactions['itemId'].isin(internal_expl_item_ids))
        ]['itemId'].nunique() # Count distinct items interacted with

        intensity = num_interactions / explanation_size if explanation_size > 0 else 0.0
        intensities.append(intensity)

    return intensities


# --- Candidate Ranking Helpers ---

def calculate_popularity_mask(
    item_ids: List[Union[int, str]],
    data_reader: DataReader
) -> Dict[Union[int, str], float]:
    """
    Calculates normalized popularity (interaction count) for items.

    Args:
        item_ids (List[Union[int, str]]): List of original item IDs.
        data_reader (DataReader): Data reader instance.

    Returns:
        Dict[Union[int, str], float]: Map from original item ID to normalized popularity [0, 1].
    """
    popularity_mask = {}
    raw_popularity = {}
    internal_item_ids = data_reader.get_new_item_ids(item_ids) # Get internal IDs for lookup
    interactions = data_reader.dataset

    if not internal_item_ids: return {}

    # Get interaction counts (popularity) for relevant items
    item_counts = interactions[interactions['itemId'].isin(internal_item_ids)]['itemId'].value_counts()

    # Map counts back to original IDs and store raw counts
    for original_item_id in item_ids:
        internal_id = data_reader.get_new_item_id(original_item_id)
        if internal_id is not None:
            raw_popularity[original_item_id] = item_counts.get(internal_id, 0)
        else:
            raw_popularity[original_item_id] = 0 # Item not found

    pop_values = list(raw_popularity.values())
    if not pop_values: return {}

    # Normalize counts to [0, 1]
    min_pop = min(pop_values)
    max_pop = max(pop_values)
    pop_range = max_pop - min_pop

    if pop_range == 0: # All items have the same popularity
        # Assign a neutral score (e.g., 0.5) or handle as needed
        normalized_value = 0.5
        for item_id in raw_popularity:
            popularity_mask[item_id] = normalized_value
    else:
        # Apply min-max normalization
        for item_id, pop_count in raw_popularity.items():
             popularity_mask[item_id] = (pop_count - min_pop) / pop_range

    return popularity_mask


def calculate_relevance_mask(
    target_item_id: Union[int, str],
    member_predictions: Dict[Union[int, str], Dict[Union[int, str], float]]
) -> Dict[Union[int, str], float]:
    """
    Creates a map of user IDs to their predicted score for the target item.

    Args:
        target_item_id (Union[int, str]): The original ID of the target item.
        member_predictions (Dict[Union[int, str], Dict[Union[int, str], float]]):
            Predictions per user: {user_id: {item_id: score, ...}}.

    Returns:
        Dict[Union[int, str], float]: Map from original user ID to their predicted score
                                     for the target item (0 if not predicted).
    """
    relevance_mask = {}
    for user_id, predictions in member_predictions.items():
        relevance_mask[user_id] = predictions.get(target_item_id, 0.0) # Default to 0 if not found
    return relevance_mask


def calculate_item_relevance_score(
    item_id: Union[int, str],
    group_member_ids: List[int],
    relevance_mask: Dict[Union[int, str], float],
    data_reader: DataReader
) -> float:
    """
    Calculates the average relevance score of an item based on user predictions
    for the *original target item*, considering only members who interacted with *this* item.
    Normalizes the score to [0, 1].

    Args:
        item_id (Union[int, str]): The original item ID to score.
        group_member_ids (List[int]): List of original group member IDs.
        relevance_mask (Dict[Union[int, str], float]): Map {user_id: predicted_score_for_target}.
        data_reader (DataReader): Data reader instance.

    Returns:
        float: Normalized average relevance score [0, 1].
    """
    internal_item_id = data_reader.get_new_item_id(item_id)
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)

    if internal_item_id is None or not internal_group_ids:
        return 0.0

    # Find members who actually interacted with *this* item (internal_item_id)
    interacting_member_ids_internal = data_reader.dataset[
        (data_reader.dataset['itemId'] == internal_item_id) &
        (data_reader.dataset['userId'].isin(internal_group_ids))
    ]['userId'].unique()

    total_relevance_score = 0.0
    num_interacting_members = 0

    # Sum the relevance scores (prediction for the *target* item) for the interacting members
    for internal_member_id in interacting_member_ids_internal:
         original_member_id = data_reader.get_original_user_id(internal_member_id)
         if original_member_id in relevance_mask:
             total_relevance_score += relevance_mask[original_member_id]
             num_interacting_members += 1

    if num_interacting_members == 0:
        return 0.0

    average_relevance = total_relevance_score / num_interacting_members

    # Normalize the average relevance score (assuming relevance scores are scaled 1-5)
    # Adjust minV/maxV if relevance_mask scores are in a different range
    max_v = TARGET_RATING_MAX # Use configured target max
    min_v = TARGET_RATING_MIN # Use configured target min
    value_range = max_v - min_v

    if value_range <= 0: # Avoid division by zero if min/max are same
        return 0.0 if average_relevance <= min_v else 1.0

    # Apply min-max normalization (clipping might be useful too)
    normalized_score = (average_relevance - min_v) / value_range
    return max(0.0, min(1.0, normalized_score)) # Clip to [0, 1]


def calculate_item_intensity_score(
    item_id: Union[int, str],
    group_member_ids: List[int],
    data_reader: DataReader
) -> float:
    """
    Calculates item intensity: fraction of group members who interacted with the item.

    Args:
        item_id (Union[int, str]): Original item ID.
        group_member_ids (List[int]): List of original group member IDs.
        data_reader (DataReader): Data reader instance.

    Returns:
        float: Item intensity score [0, 1].
    """
    internal_item_id = data_reader.get_new_item_id(item_id)
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)
    group_size = len(group_member_ids)

    if internal_item_id is None or not internal_group_ids or group_size == 0:
        return 0.0

    num_interactions = data_reader.dataset[
        (data_reader.dataset['itemId'] == internal_item_id) &
        (data_reader.dataset['userId'].isin(internal_group_ids))
    ]['userId'].nunique()

    return num_interactions / group_size


def calculate_average_rating_score(
    item_id: Union[int, str],
    group_member_ids: List[int],
    data_reader: DataReader
) -> float:
    """
    Calculates the average rating given to the item by interacting group members.
    Normalizes the score to [0, 1]. Assumes ratings are in [TARGET_RATING_MIN, TARGET_RATING_MAX].

    Args:
        item_id (Union[int, str]): Original item ID.
        group_member_ids (List[int]): List of original group member IDs.
        data_reader (DataReader): Data reader instance containing 'rating' column.

    Returns:
        float: Normalized average rating score [0, 1].
    """
    internal_item_id = data_reader.get_new_item_id(item_id)
    internal_group_ids = data_reader.get_new_user_ids(group_member_ids)

    if internal_item_id is None or not internal_group_ids:
        return 0.0

    # Get ratings for this item given by group members
    interactions = data_reader.dataset[
        (data_reader.dataset['itemId'] == internal_item_id) &
        (data_reader.dataset['userId'].isin(internal_group_ids))
    ]

    if interactions.empty:
        return 0.0 # No one in the group rated this item

    # Calculate average rating ONLY among those who rated it
    average_rating = interactions['rating'].mean()

    # Normalize the average rating score
    max_v = TARGET_RATING_MAX
    min_v = TARGET_RATING_MIN
    value_range = max_v - min_v

    if value_range <= 0:
        return 0.0 if average_rating <= min_v else 1.0

    normalized_score = (average_rating - min_v) / value_range
    return max(0.0, min(1.0, normalized_score)) # Clip to [0, 1]


def build_explanation_candidate_chart(
    items_rated_by_group: List[Union[int, str]],
    group_member_ids: List[int],
    relevance_mask: Dict[Union[int, str], float],
    popularity_mask: Dict[Union[int, str], float],
    data_reader: DataReader
) -> List[Union[int, str]]:
    """
    Scores items rated by the group based on multiple criteria (popularity,
    relevance, intensity, rating) and returns a sorted list of item IDs
    to be used as candidates for counterfactual explanations.

    Args:
        items_rated_by_group (List[Union[int, str]]): Original IDs of items rated by the group.
        group_member_ids (List[int]): Original IDs of group members.
        relevance_mask (Dict[Union[int, str], float]): Precomputed relevance scores.
        popularity_mask (Dict[Union[int, str], float]): Precomputed popularity scores.
        data_reader (DataReader): Data reader instance.

    Returns:
        List[Union[int, str]]: Sorted list of original item IDs (potential explanation candidates).
    """
    chart_scores = {}

    for item_id in items_rated_by_group:
        pop_score = popularity_mask.get(item_id, 0.0)
        item_intensity_score = calculate_item_intensity_score(item_id, group_member_ids, data_reader)
        avg_rating_score = calculate_average_rating_score(item_id, group_member_ids, data_reader)
        relevance_score = calculate_item_relevance_score(item_id, group_member_ids, relevance_mask, data_reader)

        # Combine scores (simple summation, weights could be added)
        total_score = pop_score + relevance_score + item_intensity_score + avg_rating_score
        chart_scores[item_id] = total_score

    # Sort items by the combined score in descending order
    sorted_chart = dict(sorted(chart_scores.items(), key=itemgetter(1), reverse=True))
    sorted_item_ids = list(sorted_chart.keys())

    # print(f"Chart scores (Top 15): {dict(itertools.islice(sorted_chart.items(), 15))}")
    # print(f"Sorted item IDs (Top 15): {sorted_item_ids[:15]}")

    return sorted_item_ids


# --- Main Experiment ---

def run_explanation_experiment():
    """Main function to run the counterfactual explanation experiment."""

    print("Starting explanation experiment...")
    print("--- CRITICAL WARNING ---")
    print("This script involves retraining the recommendation model potentially")
    print("thousands of times per group. This is EXTREMELY computationally")
    print("expensive and likely infeasible for large datasets or complex models.")
    print("Consider alternative approaches like model approximations, incremental")
    print("updates, influence functions, or different explanation methods for")
    print("practical applications.")
    print("-------------------------")

    # 1. Load and Prepare Initial Data
    print("Loading and preparing initial data...")
    try:
        # Assuming cfg is properly loaded elsewhere
        # data = DataReader(**cfg.data.test) # Replace placeholder if using actual library
        data = DataReader(filepath_or_buffer='dummy') # Using placeholder
        data.make_consecutive_ids_in_dataset()
        # Binarize if needed for the model (check ALS requirements)
        # data.binarize(binary_threshold=BINARY_THRESHOLD)
        print(f"Initial data shape: {data.dataset.shape}")
    except Exception as e:
        print(f"Error loading initial data: {e}")
        sys.exit(1)


    # 2. Train Initial Recommendation Model
    print("Training initial ALS model...")
    try:
        # algo = ALS(**cfg.model.als) # Replace placeholder
        algo = ALS(factors=ALS_FACTORS, regularization=ALS_REGULARIZATION, iterations=ALS_ITERATIONS) # Using placeholder
        algo.fit(data)
        print("Initial model training complete.")
    except Exception as e:
        print(f"Error training initial model: {e}")
        sys.exit(1)


    # 3. Load Groups and Prepare Item IDs
    print("Loading groups...")
    try:
        all_groups_strings = read_groups(GROUPS_FILE_PATH)
        print(f"Loaded {len(all_groups_strings)} groups.")
    except FileNotFoundError:
        print(f"Could not find groups file at {GROUPS_FILE_PATH}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading groups file: {e}")
        sys.exit(1)

    # Get all unique INTERNAL item IDs from the initial data
    all_internal_item_ids = data.dataset["itemId"].unique()
    print(f"Total unique items: {len(all_internal_item_ids)}")
    print(f"Total unique users in initial data: {data.dataset['userId'].nunique()}")


    # 4. Process Each Group
    for group_index, group_string in enumerate(all_groups_strings):
        print(f"\n--- Processing Group {group_index + 1}/{len(all_groups_strings)}: {group_string} ---")
        member_ids = get_group_members(group_string)
        print(f"Member IDs: {member_ids}")

        if not member_ids:
             print("Skipping group with no valid members.")
             continue

        # --- Initial Recommendation ---
        print("Generating initial recommendations...")
        candidate_internal_items = get_items_for_recommendation(all_internal_item_ids, member_ids, data)
        # print(f"Items to predict for (internal IDs count): {len(candidate_internal_items)}")

        initial_member_predictions = {}
        for member_id in member_ids:
            user_predictions = generate_recommendations_for_user(
                algo, member_id, candidate_internal_items, data # Use initial algo and data
            )
            initial_member_predictions[member_id] = user_predictions

        # Get the original top-1 group recommendation (the target item)
        original_target_item = get_group_recommendations(
            initial_member_predictions,
            top_n=1
        )

        if original_target_item is None:
             print("Could not determine initial target recommendation for the group. Skipping.")
             continue
        print(f"Initial Top-1 Group Recommendation (Target Item): {original_target_item}")

        # --- Prepare for Explanation Search ---
        # Get items rated by group (original IDs)
        items_rated_by_group_original = get_rated_items_by_group_members(member_ids, data)
        if not items_rated_by_group_original:
             print("Group members have not rated any items in the dataset. Cannot find explanation. Skipping.")
             continue

        # Create masks for ranking explanation candidates
        popularity_mask = calculate_popularity_mask(items_rated_by_group_original, data)
        relevance_mask = calculate_relevance_mask(original_target_item, initial_member_predictions)

        # Build the sorted list of candidate items (original IDs) for the explanation
        explanation_candidate_chart = build_explanation_candidate_chart(
            items_rated_by_group_original, member_ids, relevance_mask, popularity_mask, data
        )

        if not explanation_candidate_chart:
            print("Could not build explanation candidate chart (no rated items found?). Skipping.")
            continue

        # Determine window size (static or percentage)
        current_window_size = WINDOW_SIZE
        # Or: current_window_size = max(1, math.floor(len(explanation_candidate_chart) * WINDOW_PERCENTAGE))

        print(f"Explanation candidate chart size: {len(explanation_candidate_chart)}")
        print(f"Using sliding window size: {current_window_size}")

        # --- Sliding Window Explanation Search ---
        sliding_window = SlidingWindow(explanation_candidate_chart, current_window_size)
        found_explanation = False
        recommender_calls_count = 0 # Count how many times the *group* recommender is called after retraining
        checked_subsets_count = 0 # Count checks inside a window

        while recommender_calls_count < MAX_RECOMMENDER_CALLS:
            window_items = sliding_window.get_next_window()

            if window_items is None:
                print("Sliding window finished. No explanation found within call limit.")
                break # Exit while loop for this group

            recommender_calls_count += 1
            print(f"\n[Call {recommender_calls_count}] Checking window: {window_items}")

            # --- Retrain model excluding window_items ---
            print("  Retraining model (excluding window items)...")
            try:
                 # Create modified data for retraining
                 changed_df_window = change_data(data, member_ids, window_items)
                 # Create new DataReader and ALS instances for this retraining step
                 data_retrained_window = DataReader(dataframe=changed_df_window) # Use placeholder
                 data_retrained_window.make_consecutive_ids_in_dataset() # Important! Remap IDs
                 # data_retrained_window.binarize(BINARY_THRESHOLD) # If needed

                 # Check if data is empty after change
                 if data_retrained_window.dataset.empty:
                     print("  Warning: Dataset became empty after removing window items. Skipping window.")
                     continue

                 algo_retrained_window = ALS(factors=ALS_FACTORS, regularization=ALS_REGULARIZATION, iterations=ALS_ITERATIONS) # Placeholder
                 algo_retrained_window.fit(data_retrained_window)

                 # Generate new recommendations with the retrained model
                 retrained_member_predictions_window = {}
                 # Need to get candidate items based on the *retrained* data reader's perspective
                 # This is tricky: should candidates change? Assuming original candidates for now.
                 # Re-calculate candidate items based on the *changed* data might be more correct but adds complexity.
                 # Using original candidate_internal_items but predict with data_retrained_window
                 candidate_items_for_retrained = get_items_for_recommendation(
                       data_retrained_window.dataset['itemId'].unique(), # Use items present in retrained data
                       member_ids,
                       data_retrained_window
                 )

                 for member_id in member_ids:
                     user_preds_retrained = generate_recommendations_for_user(
                         algo_retrained_window, member_id, candidate_items_for_retrained, data_retrained_window
                     )
                     retrained_member_predictions_window[member_id] = user_preds_retrained

                 group_rec_list_window = get_group_recommendations(
                     retrained_member_predictions_window,
                     top_n=RECOMMENDATION_LIST_SIZE # Get top N list
                 )
                 print(f"  New Top-{RECOMMENDATION_LIST_SIZE} Recs (after removing window): {group_rec_list_window}")

            except Exception as e:
                 print(f"  Error during retraining or prediction for window {window_items}: {e}")
                 continue # Skip to the next window

            # --- Check if target item disappeared ---
            if original_target_item not in group_rec_list_window:
                print(f"  Target item {original_target_item} removed! Searching minimal subset in window...")
                # Target item removed by the whole window, now find the minimal subset responsible

                minimal_explanation_found_in_window = False
                for subset_size in range(1, len(window_items) + 1):
                    if minimal_explanation_found_in_window: break # Stop if minimal found

                    for explanation_candidate in itertools.combinations(window_items, subset_size):
                        explanation_candidate = list(explanation_candidate)
                        checked_subsets_count += 1

                        # Avoid infinite loops if subset check also counts towards main limit
                        if recommender_calls_count + checked_subsets_count > MAX_RECOMMENDER_CALLS:
                            print("  Max call limit reached during subset check.")
                            minimal_explanation_found_in_window = True # Break outer loops
                            break

                        print(f"    [Sub-check {checked_subsets_count}] Testing subset: {explanation_candidate}")

                        # --- Retrain model excluding subset ---
                        # print(f"      Retraining model (excluding subset)...") # Verbose
                        try:
                            changed_df_subset = change_data(data, member_ids, explanation_candidate)
                            data_retrained_subset = DataReader(dataframe=changed_df_subset) # Placeholder
                            data_retrained_subset.make_consecutive_ids_in_dataset() # Remap IDs
                            # data_retrained_subset.binarize(BINARY_THRESHOLD) # If needed

                            if data_retrained_subset.dataset.empty:
                                 print("      Warning: Dataset became empty after removing subset. Skipping subset.")
                                 continue

                            algo_retrained_subset = ALS(factors=ALS_FACTORS, regularization=ALS_REGULARIZATION, iterations=ALS_ITERATIONS) # Placeholder
                            algo_retrained_subset.fit(data_retrained_subset)

                            retrained_member_predictions_subset = {}
                            candidate_items_for_subset = get_items_for_recommendation(
                                 data_retrained_subset.dataset['itemId'].unique(),
                                 member_ids,
                                 data_retrained_subset
                            )

                            for member_id in member_ids:
                                user_preds_subset = generate_recommendations_for_user(
                                    algo_retrained_subset, member_id, candidate_items_for_subset, data_retrained_subset
                                )
                                retrained_member_predictions_subset[member_id] = user_preds_subset

                            group_rec_list_subset = get_group_recommendations(
                                retrained_member_predictions_subset,
                                top_n=RECOMMENDATION_LIST_SIZE
                            )
                            # print(f"      New Top-{RECOMMENDATION_LIST_SIZE} Recs (after removing subset): {group_rec_list_subset}")

                        except Exception as e:
                             print(f"      Error during retraining/prediction for subset {explanation_candidate}: {e}")
                             continue # Skip to next subset

                        # --- Check if target item is still removed by this subset ---
                        if original_target_item not in group_rec_list_subset:
                            print("-" * 20)
                            print(f"Minimal Counterfactual Explanation Found for Group {group_string}!")
                            print(f"Explanation (Items to remove): {explanation_candidate}")
                            print(f"Original Target Item: {original_target_item}")
                            new_top_item = group_rec_list_subset[0] if group_rec_list_subset else "None"
                            print(f"New Top Item (after removing explanation): {new_top_item}")
                            print(f"Found at total recommender calls: {recommender_calls_count} (window) + {checked_subsets_count} (subset checks)")
                            print(f"Explanation Size: {len(explanation_candidate)}")

                            # Calculate metrics for the found explanation
                            avg_item_intensity = calculate_average_item_intensity(explanation_candidate, member_ids, data)
                            user_intensity = calculate_user_intensity(explanation_candidate, member_ids, data)
                            print(f"Average Item Intensity per item in explanation: {[f'{x:.3f}' for x in avg_item_intensity]}")
                            print(f"User Intensity per user in group: {[f'{x:.3f}' for x in user_intensity]}")
                            print("-" * 20)

                            found_explanation = True
                            minimal_explanation_found_in_window = True
                            break # Stop searching combinations for this window

            if found_explanation:
                break # Exit the while loop for this group

        # End of while loop for the current group
        if not found_explanation:
            print(f"Could not find counterfactual explanation for group {group_string} within {MAX_RECOMMENDER_CALLS} calls.")

    print("\nExperiment finished.")

# --- Run the experiment ---
if __name__ == "__main__":
    run_explanation_experiment()