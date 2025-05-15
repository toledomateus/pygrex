import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from recoxplainer.data_reader.data_reader import DataReader


@pytest.fixture
def valid_df():
    """Fixture providing a valid test DataFrame."""
    return pd.DataFrame(
        {
            "userId": [1, 2, 3, 1],
            "itemId": [100, 200, 300, 400],
            "rating": [4.5, 3.0, 5.0, 2.0],
            "timestamp": [1000, 2000, 3000, 4000],
        }
    )


@pytest.fixture
def test_csv_path(valid_df):
    """Fixture providing a temporary CSV file with valid data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_csv_path = os.path.join(temp_dir, "test_data.csv")
        valid_df.to_csv(test_csv_path, sep=",", index=False)
        yield test_csv_path


def test_init_with_dataframe(valid_df):
    """Test initialization with a DataFrame."""
    reader = DataReader(dataframe=valid_df)
    pd.testing.assert_frame_equal(reader.dataset, valid_df)
    assert reader.num_user == 3
    assert reader.num_item == 4


def test_init_with_filepath(test_csv_path):
    """Test initialization with filepath."""
    reader = DataReader(
        filepath_or_buffer=test_csv_path,
        sep=",",
        names=["userId", "itemId", "rating", "timestamp"],
        skiprows=1,  # Skip header
    )
    # Check that dataset was loaded correctly
    assert reader.num_user == 3
    assert reader.num_item == 4


def test_dataset_loading_file_not_found():
    """Test exception when file is not found."""
    with pytest.raises(FileNotFoundError):
        DataReader(
            filepath_or_buffer="nonexistent.csv",
            sep=",",
            names=["userId", "itemId", "rating", "timestamp"],
        )


def test_dataset_validation_invalid_columns():
    """Test validation for invalid columns."""
    # Create a DataFrame with non-numeric userId
    invalid_df = pd.DataFrame(
        {
            "userId": ["user1", "user2", "user3"],
            "itemId": [100, 200, 300],
            "rating": [4.5, 3.0, 5.0],
            "timestamp": [1000, 2000, 3000],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        DataReader(dataframe=invalid_df)
    assert "cannot be converted to a numeric type" in str(excinfo.value)


def test_dataset_validation_empty():
    """Test validation for empty DataFrame."""
    empty_df = pd.DataFrame(columns=["userId", "itemId", "rating", "timestamp"])
    with pytest.raises(ValueError) as excinfo:
        DataReader(dataframe=empty_df)
    assert "DataFrame cannot be empty" in str(excinfo.value)


def test_dataset_validation_invalid_types():
    """Test validation for invalid data types."""
    # Create DataFrame with invalid rating (string)
    invalid_types_df = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "itemId": [100, 200, 300],
            "rating": ["high", "medium", "low"],
            "timestamp": [1000, 2000, 3000],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        DataReader(dataframe=invalid_types_df)
    assert "cannot be converted to a numeric type" in str(excinfo.value)


def test_dataset_validation_missing_values():
    """Test validation for missing values."""
    # Create DataFrame with NaN values
    missing_values_df = pd.DataFrame(
        {
            "userId": [1, 2, np.nan],
            "itemId": [100, 200, 300],
            "rating": [4.5, 3.0, 5.0],
            "timestamp": [1000, 2000, 3000],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        DataReader(dataframe=missing_values_df)
    assert "contains missing values" in str(excinfo.value)


def test_get_raw_dataset_not_set():
    """Test get_raw_dataset when not set."""
    # Mock pandas.read_csv to raise FileNotFoundError
    with patch(
        "pandas.read_csv", side_effect=FileNotFoundError("File not found: dummy.csv")
    ):
        with pytest.raises(FileNotFoundError):
            reader = DataReader(
                filepath_or_buffer="dummy.csv",
                sep=",",
                names=["userId", "itemId", "rating", "timestamp"],
            )


def test_make_consecutive_ids_in_dataset(valid_df):
    """Test making IDs consecutive."""
    # Create a DataFrame with non-consecutive IDs
    df = pd.DataFrame(
        {
            "userId": [100, 200, 300, 100],
            "itemId": [500, 600, 700, 800],
            "rating": [4.5, 3.0, 5.0, 2.0],
            "timestamp": [1000, 2000, 3000, 4000],
        }
    )

    reader = DataReader(dataframe=df)
    reader.make_consecutive_ids_in_dataset()

    # Check that IDs are now consecutive
    assert set(reader.dataset["userId"]) == {0, 1, 2}
    assert set(reader.dataset["itemId"]) == {0, 1, 2, 3}

    # Check mapping
    assert reader.get_original_user_id(0) == 100
    assert reader.get_original_item_id(0) == 500


def test_make_consecutive_ids_not_set():
    """Test make_consecutive_ids_in_dataset when dataset is not set."""
    with patch(
        "pandas.read_csv", side_effect=FileNotFoundError("File not found: dummy.csv")
    ):
        with pytest.raises(FileNotFoundError):
            reader = DataReader(
                filepath_or_buffer="dummy.csv",
                sep=",",
                names=["userId", "itemId", "rating", "timestamp"],
            )


def test_binarize(valid_df):
    """Test binarization of ratings."""
    reader = DataReader(dataframe=valid_df)
    reader.binarize(binary_threshold=3.5)

    # Check that ratings are binarized
    expected = [1, 0, 1, 0]  # 4.5 > 3.5, 3.0 <= 3.5, 5.0 > 3.5, 2.0 <= 3.5
    assert list(reader.dataset["rating"]) == expected


def test_binarize_not_set():
    """Test binarize when dataset is not set."""
    with patch(
        "pandas.read_csv", side_effect=FileNotFoundError("File not found: dummy.csv")
    ):
        with pytest.raises(FileNotFoundError):
            reader = DataReader(
                filepath_or_buffer="dummy.csv",
                sep=",",
                names=["userId", "itemId", "rating", "timestamp"],
            )


def test_get_original_user_id_not_set(valid_df):
    """Test get_original_user_id when mapping is not set."""
    # Create reader without calling make_consecutive_ids_in_dataset
    reader = DataReader(dataframe=valid_df)
    with pytest.raises(ValueError) as excinfo:
        reader.get_original_user_id(0)
    assert "ID mapping not set" in str(excinfo.value)
