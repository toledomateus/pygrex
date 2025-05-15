from typing import List, Optional, Union
import numpy as np
import pandas as pd
import warnings
import os


class DataReader:
    def __init__(
        self,
        filepath_or_buffer: Optional[str] = None,
        sep: Optional[str] = None,
        names: Optional[List[str]] = None,
        skiprows: int = 0,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the DataReader with either a DataFrame or file parameters.

        Args:
            filepath_or_buffer (Optional[str]): Path to the CSV file or buffer.
            sep (Optional[str]): Separator used in the CSV file.
            names (Optional[List[str]]): List of column names for the CSV file.
            skiprows (int, optional): Number of rows to skip in the CSV file. Defaults to 0.
            dataframe (Optional[pd.DataFrame], optional): A DataFrame to use directly. Defaults to None.

        Raises:
            ValueError: If neither `dataframe` nor valid file parameters are provided.
            FileNotFoundError: If the file cannot be found when loading from file.
            pd.errors.ParserError: If the CSV file cannot be parsed when loading from file.

        Note:
            If `dataframe` is provided, it takes precedence, and file-related parameters
            are ignored but stored for reference. A warning is issued in this case.
            The DataFrame must contain columns: 'userId', 'itemId', 'rating', 'timestamp'.
        """
        if dataframe is None and (not filepath_or_buffer or not sep or not names):
            raise ValueError(
                "Must provide either a DataFrame or valid file parameters."
            )

        self.filepath_or_buffer = filepath_or_buffer
        self.sep = sep
        self.names = names
        self.skiprows = skiprows
        self._dataset = None
        self._raw_dataset = None
        self._num_user: Optional[int] = None
        self._num_item: Optional[int] = None
        self.original_user_id: Optional[pd.DataFrame] = None
        self.original_item_id: Optional[pd.DataFrame] = None
        self.new_user_id: Optional[pd.DataFrame] = None
        self.new_item_id: Optional[pd.DataFrame] = None

        if dataframe is not None:
            if any(param is not None for param in [filepath_or_buffer, sep, names]):
                warnings.warn(
                    "DataFrame provided; file parameters (filepath_or_buffer, sep, names) are ignored.",
                    UserWarning,
                )
            self.dataset = dataframe

        elif filepath_or_buffer and sep and names:
            # Eagerly load data if file parameters are provided
            try:
                loaded_df = pd.read_csv(
                    filepath_or_buffer=self.filepath_or_buffer,
                    sep=self.sep,
                    names=self.names,
                    skiprows=self.skiprows,
                    engine="python",
                )
                self._raw_dataset = loaded_df.copy()
                # Use the setter to handle dataset validation and setting _num_user/_num_item
                self.dataset = loaded_df
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {self.filepath_or_buffer}")
            except pd.errors.ParserError as e:
                raise pd.errors.ParserError(f"Failed to parse CSV: {str(e)}")
        else:
            raise ValueError(
                "Must provide either a DataFrame or valid file parameters."
            )

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Get the dataset DataFrame.
        """
        if self._dataset is None:
            if self._dataset is None:
                # If it reach here and _dataset is None, it means initialization failed
                # or an empty DataFrame was set.
                # This state should ideally not be reached with eager loading if file params were valid.
                raise ValueError("Dataset is not loaded or is not valid.")
        return self._dataset

    @dataset.setter
    def dataset(self, new_data: pd.DataFrame) -> None:
        """
        Set the dataset and compute the number of unique users and items.

        Args:
            new_data (pd.DataFrame): The new dataset to set.

        Raises:
            ValueError: If the DataFrame is None, empty, lacks required columns,
                       or contains invalid data types/missing values.
        """
        if new_data is None:
            raise ValueError("DataFrame cannot be None")
        if new_data.empty:
            raise ValueError("DataFrame cannot be empty")

        # Validate data types
        for col in ["userId", "itemId", "rating"]:
            if not pd.api.types.is_numeric_dtype(new_data[col]):
                warnings.warn(
                    f"Column '{col}' is not numeric. Attempting conversion.",
                    UserWarning,
                )
                try:
                    new_data[col] = pd.to_numeric(new_data[col])
                except ValueError:
                    raise ValueError(
                        f"Column '{col}' cannot be converted to a numeric type."
                    )

        # Check for missing values in essential columns
        if new_data[["userId", "itemId", "rating"]].isnull().any().any():
            raise ValueError(
                "DataFrame contains missing values in essential columns (userId, itemId, rating)."
            )

        self._dataset = new_data
        self._raw_dataset = new_data.copy()
        self._num_user = int(self._dataset["userId"].nunique())
        self._num_item = int(self._dataset["itemId"].nunique())
        # Set the index to userId and itemId for easier access
        # Reset id mappings as they are now invalid for the new dataset
        self.original_user_id = None
        self.original_item_id = None
        self.new_user_id = None
        self.new_item_id = None

    def get_raw_dataset(self) -> pd.DataFrame:
        """
        Get the raw dataset as loaded from the file or initially set.

        Returns:
            pd.DataFrame: The raw dataset.

        Raises:
            ValueError: If the raw dataset is not set.
        """
        if self._raw_dataset is None:
            raise ValueError(
                "Raw dataset is not set. Load data from file or set a DataFrame first."
            )
        return self._raw_dataset

    @staticmethod
    def _create_id_mapping(column: pd.Series, new_column_name: str) -> pd.DataFrame:
        """
        Create a mapping for consecutive IDs.

        Args:
            column (pd.Series): The column to map.
            new_column_name (str): The name of the new column for consecutive IDs.

        Returns:
            pd.DataFrame: A DataFrame with the original and mapped IDs.

        Raises:
            ValueError: If the column is empty.
        """
        if column.empty:
            raise ValueError("Cannot create ID mapping for an empty column")
        unique_values = column.drop_duplicates().reset_index(drop=True)
        mapping = pd.DataFrame(
            {column.name: unique_values, new_column_name: np.arange(len(unique_values))}
        )
        return mapping

    def make_consecutive_ids_in_dataset(self) -> None:
        """
        Map user and item IDs to consecutive integers starting from 0.

        Modifies the dataset in-place and stores mappings for original and new IDs.

        Example:
            Original user IDs [100, 200, 300] -> New user IDs [0, 1, 2]

        Raises:
            ValueError: If the dataset is not set.
        """
        if self._dataset is None:
            raise ValueError("Dataset must be loaded or set before mapping IDs")

        # Check if IDs are already consecutive
        if (
            self._dataset["userId"].min() == 0
            and self._dataset["userId"].max() == self._num_user - 1
            and self._dataset["itemId"].min() == 0
            and self._dataset["itemId"].max() == self._num_item - 1
        ):
            print("IDs already appear to be consecutive. Skipping mapping.")
            # Populate mapping tables even if IDs are already consecutive
            self.original_user_id = (
                self._dataset[["userId"]]
                .drop_duplicates()
                .set_index("userId")
                .sort_index()
            )
            self.original_user_id["new_id"] = self.original_user_id.index
            self.new_user_id = (
                self.original_user_id.copy()
                .rename(columns={"new_id": "original_id"})
                .set_index("new_id")
                .sort_index()
            )

            self.original_item_id = (
                self._dataset[["itemId"]]
                .drop_duplicates()
                .set_index("itemId")
                .sort_index()
            )
            self.original_item_id["new_id"] = self.original_item_id.index
            self.new_item_id = (
                self.original_item_id.copy()
                .rename(columns={"new_id": "original_id"})
                .set_index("new_id")
                .sort_index()
            )

            return  # Exit if IDs are already consecutive

        dataset = self.dataset.copy()

        # Create user ID mapping
        user_id_mapping = self._create_id_mapping(dataset["userId"], "new_userId")
        dataset["userId"] = dataset["userId"].map(
            user_id_mapping.set_index("userId")["new_userId"]
        )

        # Create item ID mapping
        item_id_mapping = self._create_id_mapping(dataset["itemId"], "new_itemId")
        dataset["itemId"] = dataset["itemId"].map(
            item_id_mapping.set_index("itemId")["new_itemId"]
        )

        # Store mappings
        self.original_user_id = user_id_mapping.set_index("new_userId")
        self.original_item_id = item_id_mapping.set_index("new_itemId")
        self.new_user_id = user_id_mapping.set_index("userId")
        self.new_item_id = item_id_mapping.set_index("itemId")

        # Update dataset with mapped IDs
        dataset["userId"] = dataset["userId"].astype(int)
        dataset["itemId"] = dataset["itemId"].astype(int)
        self._dataset = dataset

    def binarize(
        self, binary_threshold: float = 1, inplace: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Binarize ratings into 0 or 1 based on a threshold (implicit feedback).

        Args:
            binary_threshold (float, optional): Threshold for binarization. Defaults to 1.0.
            inplace (bool, optional): If True, modify the dataset in-place. If False, return a new DataFrame.
                                     Defaults to True.

        Returns:
            Optional[pd.DataFrame]: The binarized dataset if inplace=False, else None.

        Raises:
            ValueError: If the dataset is not set or binary_threshold is invalid.

        Example:
            Ratings [0.5, 2.0, 3.0] with threshold=1.0 -> [0, 1, 1]
        """
        if self._dataset is None:
            raise ValueError("Dataset must be loaded or set before binarization")
        if not isinstance(binary_threshold, (int, float)):
            raise ValueError("binary_threshold must be a number")

        dataset = self._dataset if inplace else self._dataset.copy()
        dataset["rating"] = (dataset["rating"] > binary_threshold).astype(int)

        if not inplace:
            return dataset
        self._dataset = dataset
        return None

    @property
    def num_user(self) -> int:
        """
        Get the number of unique users.

        Returns:
            int: Number of unique users.

        Raises:
            ValueError: If the dataset is not set.
        """
        if self._num_user is None:
            raise ValueError("Dataset must be loaded or set to compute num_user")
        return self._num_user

    @property
    def num_item(self) -> int:
        """
        Get the number of unique items.

        Returns:
            int: Number of unique items.

        Raises:
            ValueError: If the dataset is not set.
        """
        if self._num_item is None:
            raise ValueError("Dataset must be loaded or set to compute num_item")
        return self._num_item

    def get_original_user_id(self, u: Union[int, List[int]]) -> Union[int, List[int]]:
        """
        Get the original user ID(s) from the new (consecutive) ID(s).

        Args:
            u (Union[int, List[int]]): New user ID(s).

        Returns:
            Union[int, List[int]]: Original user ID(s).

        Raises:
            ValueError: If ID mapping is not set or if any ID is not found.
        """
        if self.original_user_id is None:
            raise ValueError(
                "ID mapping not set. Call make_consecutive_ids_in_dataset first"
            )
        try:
            if isinstance(u, (int, np.integer)):
                return int(self.original_user_id.loc[u, "userId"])
            return list(self.original_user_id.loc[u, "userId"])
        except KeyError as e:
            raise ValueError(f"User ID(s) not found: {e}")

    def get_original_item_id(self, i: Union[int, List[int]]) -> Union[int, List[int]]:
        """
        Get the original item ID(s) from the new (consecutive) ID(s).

        Args:
            i (Union[int, List[int]]): New item ID(s).

        Returns:
            Union[int, List[int]]: Original item ID(s).

        Raises:
            ValueError: If ID mapping is not set or if any ID is not found.
        """
        if self.original_item_id is None:
            raise ValueError(
                "ID mapping not set. Call make_consecutive_ids_in_dataset first"
            )
        try:
            if isinstance(i, (int, np.integer)):
                return int(self.original_item_id.loc[i, "itemId"])
            return list(self.original_item_id.loc[i, "itemId"])
        except KeyError as e:
            raise ValueError(f"Item ID(s) not found: {e}")

    def get_new_user_id(self, u: Union[int, List[int]]) -> Union[int, List[int]]:
        """
        Get the new (consecutive) user ID(s) from the original ID(s).

        Args:
            u (Union[int, List[int]]): Original user ID(s).

        Returns:
            Union[int, List[int]]: New user ID(s).

        Raises:
            ValueError: If ID mapping is not set or if any ID is not found.
        """
        if self.new_user_id is None:
            raise ValueError(
                "ID mapping not set. Call make_consecutive_ids_in_dataset first"
            )
        try:
            if isinstance(u, (int, np.integer)):
                return int(self.new_user_id.loc[u, "new_userId"])
            return list(self.new_user_id.loc[u, "new_userId"])
        except KeyError as e:
            raise ValueError(f"User ID(s) not found: {e}")

    def get_new_item_id(self, i: Union[int, List[int]]) -> Union[int, List[int]]:
        """
        Get the new (consecutive) item ID(s) from the original ID(s).

        Args:
            i (Union[int, List[int]]): Original item ID(s).

        Returns:
            Union[int, List[int]]: New item ID(s).

        Raises:
            ValueError: If ID mapping is not set or if any ID is not found.
        """
        if self.new_item_id is None:
            raise ValueError(
                "ID mapping not set. Call make_consecutive_ids_in_dataset first"
            )
        try:
            if isinstance(i, (int, np.integer)):
                return int(self.new_item_id.loc[i, "new_itemId"])
            return list(self.new_item_id.loc[i, "new_itemId"])
        except KeyError as e:
            raise ValueError(f"Item ID(s) not found: {e}")
