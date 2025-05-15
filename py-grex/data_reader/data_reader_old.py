from typing import List, Optional
import numpy as np
import pandas as pd
import os


class DataReader:
    def __init__(
        self,
        filepath_or_buffer: str,
        sep: str,
        names: list,
        skiprows: int = 0,
        dataframe: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the DataReader with either a DataFrame or file parameters.

        Args:
            filepath_or_buffer (str): Path to the CSV file or buffer.
            sep (str): Separator used in the CSV file.
            names (list): List of column names for the CSV file.
            skiprows (int, optional): Number of rows to skip in the CSV file. Defaults to 0.
            dataframe (Optional[pd.DataFrame], optional): A DataFrame to use directly. Defaults to None.

        Note:
            If `dataframe` is provided, it takes precedence, and file-related parameters
            (`filepath_or_buffer`, `sep`, `names`, `skiprows`) are ignored but can still be passed.
            The DataFrame must contain columns: 'userId', 'itemId', 'rating', 'timestamp'.
        """
        if dataframe is None:
            self.filepath_or_buffer = filepath_or_buffer
            self.sep = sep
            self.names = names
            self.skiprows = skiprows
            self._dataset = None
        else:
            self.dataset = dataframe
            self.filepath_or_buffer = filepath_or_buffer
            self.sep = sep
            self.names = names
            self.skiprows = skiprows

        self._num_user = None
        self._num_item = None
        self.dataset

    @property
    def dataset(self):
        """
        Get the dataset, loading it from file if not already set.

        Returns:
            pd.DataFrame: The dataset with user-item interactions.
        """
        if self._dataset is None:
            self._dataset = pd.read_csv(
                filepath_or_buffer=self.filepath_or_buffer,
                sep=self.sep,
                names=self.names,
                skiprows=self.skiprows,
                engine="python",
            )

        self._num_user = int(self._dataset["userId"].nunique())
        self._num_item = int(self._dataset["itemId"].nunique())
        return self._dataset

    @dataset.setter
    def dataset(self, new_data):
        """
        Set the dataset and compute the number of unique users and items.

        Args:
            new_data (pd.DataFrame): The new dataset to set.

        Raises:
            ValueError: If the DataFrame lacks required columns.
        """
        if new_data is None:
            raise ValueError("DataFrame cannot be None")
        # Validate required columns
        required_columns = {"userId", "itemId", "rating", "timestamp"}
        if not required_columns.issubset(new_data.columns):
            raise ValueError(f"DataFrame must have columns: {required_columns}")
        self._dataset = new_data

    @staticmethod
    def _create_id_mapping(column: pd.Series, new_column_name: str) -> pd.DataFrame:
        """
        Create a mapping for consecutive IDs.

        Args:
            column (pd.Series): The column to map.
            new_column_name (str): The name of the new column for consecutive IDs.

        Returns:
            pd.DataFrame: A DataFrame with the original and mapped IDs.
        """
        unique_values = column.drop_duplicates().reset_index(drop=True)
        mapping = pd.DataFrame(
            {column.name: unique_values, new_column_name: np.arange(len(unique_values))}
        )
        return mapping

    def make_consecutive_ids_in_dataset(self):
        dataset = self.dataset.rename(
            {"userId": "user_id", "itemId": "item_id"}, axis=1
        )

        # Create user ID mapping
        user_id_mapping = self._create_id_mapping(dataset["user_id"], "userId")
        self._dataset = pd.merge(dataset, user_id_mapping, on="user_id", how="left")

        # Create item ID mapping
        item_id_mapping = self._create_id_mapping(dataset["item_id"], "itemId")
        self._dataset = pd.merge(
            self._dataset, item_id_mapping, on="item_id", how="left"
        )

        # Store mappings
        self.original_user_id = user_id_mapping.set_index("userId")
        self.original_item_id = item_id_mapping.set_index("itemId")
        self.new_user_id = user_id_mapping.set_index("user_id")
        self.new_item_id = item_id_mapping.set_index("item_id")

        # Keep only necessary columns
        self._dataset = self._dataset[["userId", "itemId", "rating", "timestamp"]]

        # Ensure IDs are integers
        self._dataset["userId"] = self._dataset["userId"].astype(int)
        self._dataset["itemId"] = self._dataset["itemId"].astype(int)

    def binarize(self, binary_threshold=1):
        """binarize into 0 or 1, imlicit feedback"""

        self._dataset.loc[self._dataset["rating"] > binary_threshold, "rating"] = 1
        self._dataset.loc[self._dataset["rating"] <= binary_threshold, "rating"] = 0

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_item

    def get_original_user_id(self, u):
        if isinstance(u, int):
            return self.original_user_id.loc[u].user_id

        return list(self.original_user_id.loc[u].user_id)

    def get_original_item_id(self, i):
        if isinstance(i, int):
            return self.original_item_id.loc[i].item_id

        return list(self.original_item_id.loc[i].item_id)

    def get_new_user_id(self, u):
        if isinstance(u, int):
            return self.new_user_id.loc[u].userId

        return list(self.new_user_id.loc[u].userId)

    def get_new_item_id(self, i):
        if isinstance(i, int):
            return self.new_item_id.loc[i].itemId

        return list(self.new_item_id.loc[i].itemId)
