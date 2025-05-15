from typing import List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

from recoxplainer.data_reader.data_reader import DataReader


class GroupInteractionHandler:
    def __init__(self, groups_filepath: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the GroupInteractionHandler.

        Args:
            groups_filepath: Path to directory containing group files or list of file paths
        """
        # Convert to Path objects
        if isinstance(groups_filepath, (str, Path)):
            path = Path(groups_filepath)
            # If a single directory path is provided, get all files in it
            if path.is_dir():
                self.groups_filepath = [
                    str(file) for file in path.iterdir() if file.is_file()
                ]
            else:
                self.groups_filepath = [str(path)]
        else:
            # If a list of paths is provided, convert all to Path and then to strings
            self.groups_filepath = [str(Path(p)) for p in groups_filepath]

    def _get_group_filepath(self, filename: str) -> str:
        """
        Get a specific group file path by matching the filename.

        Args:
            filename (str): The name of the file to search for.

        Returns:
            str: The matched file path.

        Raises:
            ValueError: Error: File does not exist
            ValueError: No file found containing '{filename}' in its name.
        """
        for path_str in self.groups_filepath:
            if filename in path_str:  # Check if filename is part of the path
                path = Path(path_str).resolve()
                if path.exists():
                    return str(path)
                else:
                    raise ValueError(f"Error: File does not exist: {path}")

        raise ValueError(f"Error: No file found containing '{filename}' in its name.")

    def read_groups(self, filename: str) -> List[str]:
        """
        Method to read group IDs from a specified file.

        Args:
            filename (str): Name of the file containing group IDs.

        Returns:
            List[str]: List of group IDs.

        Raises:
            ValueError: If groups path is not specified in configuration
        """
        if not filename:
            raise ValueError("Groups path not specified in configuration")

        filepath = self._get_group_filepath(filename)

        # Use Path for file reading
        path = Path(filepath)
        return [line.strip() for line in path.read_text().splitlines()]

    def parse_group_members(self, group: str) -> List[int]:
        """
        Parse group ID to get member IDs.

        Args:
            group: Group ID string

        Returns:
            List of member IDs
        """
        group = group.strip()
        members = group.split("_")
        return [int(m) for m in members]

    def get_group_members(self, group: Union[List[Union[int, str]], str]) -> List[int]:
        """
        Get group members from a group ID string or list.

        Args:
            group: Group ID string in format "id1_id2_id3" or list of IDs

        Returns:
            List of member IDs as integers

        Raises:
            ValueError: If any member ID cannot be converted to an integer
            TypeError: If group is neither a string nor a list
        """

        if isinstance(group, list):
            return [int(member) for member in group]

        if not isinstance(group, str):
            raise TypeError(f"Expected string or list, got {type(group).__name__}")

        group = group.strip()
        if not group:
            return []

        try:
            return [int(member) for member in group.split("_")]
        except ValueError as e:
            raise ValueError(f"Invalid member ID in group: {str(e)}")

    def create_modified_dataset(
        self,
        original_data: Union[pd.DataFrame, DataReader],
        group_ids: List[Union[int, str]],
        item_ids: List[Union[int, str]],
        data: Optional[DataReader] = None,
    ) -> pd.DataFrame:
        """
        Creates a modified dataset by removing interactions between specified groups and items.

        Args:
            original_data: Either a pandas DataFrame or a DataReader object containing the dataset
            group_ids: List of group IDs to consider for removal
            item_ids: List of item IDs to consider for removal
            data: Optional DataReader object if original_data is a DataFrame

        Returns:
            pd.DataFrame: A pandas DataFrame with the specified interactions removed

        Raises:
            ValueError: If input data types are incorrect
        """
        # Determine the data source and target dataset
        if isinstance(original_data, DataReader):
            data_reader = original_data
            dataset = original_data.dataset
        elif isinstance(original_data, pd.DataFrame) and isinstance(data, DataReader):
            data_reader = data
            dataset = original_data
        else:
            raise ValueError(
                "Either original_data must be a DataReader or data must be provided as a DataReader"
            )

        # Convert IDs to internal representation
        new_group_ids = [
            data_reader.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            for g in group_ids
        ]

        new_item_ids = [
            data_reader.get_new_item_id(
                int(i) if isinstance(i, (int, np.integer)) else i
            )
            for i in item_ids
        ]

        # Create mask for rows to keep (inverse of rows to drop)
        mask = ~(dataset.itemId.isin(new_item_ids) & dataset.userId.isin(new_group_ids))

        return dataset[mask]

    def get_rated_items_by_all_groupmembers(
        self, group: List[Union[int, str]], original_data: DataReader
    ) -> np.ndarray:
        """
        Get all items rated by any member of the group.

        Args:
            group: List of user IDs
            original_data: Data object with mapping methods

        Returns:
            np.ndarray: Array of original item IDs rated by any group member
        """
        # Convert group members to new user IDs
        new_group = [
            original_data.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            for g in group
        ]

        # Get unique items rated by any group member
        group_items = original_data.dataset[
            original_data.dataset.userId.isin(new_group)
        ]["itemId"].unique()

        # Convert back to original item IDs
        return original_data.get_original_item_id(group_items)

    def get_common_rated_items(
        self, group: List[Union[int, str]], original_data: DataReader
    ) -> np.ndarray:
        """
        Get items rated by all members of the group (intersection of rated items).

        Args:
            group: List of user IDs
            original_data: DataReader object with mapping methods

        Returns:
            np.ndarray: Array of original item IDs rated by all group members
        """
        # Convert group members to new user IDs
        new_group = [
            original_data.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            for g in group
        ]

        # Get items rated by each group member
        rated_items_per_member = []
        for user_id in new_group:
            user_items = original_data.dataset[original_data.dataset.userId == user_id][
                "itemId"
            ].unique()
            rated_items_per_member.append(set(user_items))

        # Find intersection of all rated items
        if rated_items_per_member:
            common_items = set.intersection(*rated_items_per_member)
            common_items_array = np.array(list(common_items))
            # Convert back to original item IDs
            return original_data.get_original_item_id(common_items_array)
        else:
            return np.array([])

    def get_items_for_group_recommendation(
        self, data: pd.DataFrame, item_ids: np.ndarray, group: List[int]
    ) -> np.ndarray:
        """
        Get items for group recommendation (those not interacted with by any group member).

        Args:
            data: DataFrame with interaction data
            item_ids: Array of all item IDs
            group: List of group member IDs

        Returns:
            Array of item IDs not interacted with by any group member
        """
        item_ids_group = data.loc[data.userId.isin(group), "itemId"]
        return np.setdiff1d(item_ids, item_ids_group)

    def get_group_preferences(
        self, group: List[Union[int, str]], data_reader: DataReader
    ) -> pd.DataFrame:
        """
        Get all preferences (ratings) by all members of the group.

        Args:
            group: List of user IDs
            data_reader: DataReader object with the dataset

        Returns:
            pd.DataFrame: DataFrame containing all preferences by group members
        """
        # Convert group members to new user IDs
        new_group = [
            data_reader.get_new_user_id(
                int(g) if isinstance(g, (int, np.integer)) else g
            )
            for g in group
        ]

        # Get all interactions by group members
        group_preferences = data_reader.dataset[
            data_reader.dataset.userId.isin(new_group)
        ].copy()

        return group_preferences
