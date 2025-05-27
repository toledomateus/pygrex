import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch

from pygrex.data_reader import DataReader, GroupInteractionHandler


@pytest.fixture
def test_environment():
    """Set up test environment before each test."""
    # Create a temporary directory for test files
    test_dir = Path(tempfile.mkdtemp())

    # Create test group files
    group1_file = test_dir / "group1.txt"
    group2_file = test_dir / "group2.txt"
    group1_file.write_text("1_2_3\n4_5_6\n7_8_9")
    group2_file.write_text("10_11_12\n13_14_15")

    # Create test data for DataReader mock
    test_data = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5],
            "itemId": [101, 102, 102, 103, 101, 104, 105, 106, 107],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0, 3.0],
        }
    )

    # Initialize the handler
    handler = GroupInteractionHandler(test_dir)

    yield {
        "test_dir": test_dir,
        "group1_file": group1_file,
        "group2_file": group2_file,
        "test_data": test_data,
        "handler": handler,
    }

    # Clean up test environment after each test
    shutil.rmtree(test_dir)


def test_init_with_directory(test_environment):
    """Test initialization with a directory path."""
    handler = GroupInteractionHandler(test_environment["test_dir"])
    assert len(handler.filepath_or_buffer) == 2
    file_paths = set(handler.filepath_or_buffer)
    assert str(test_environment["group1_file"]) in file_paths
    assert str(test_environment["group2_file"]) in file_paths


def test_init_with_file_list(test_environment):
    """Test initialization with a list of file paths."""
    file_list = [test_environment["group1_file"], test_environment["group2_file"]]
    handler = GroupInteractionHandler(file_list)
    assert len(handler.filepath_or_buffer) == 2
    assert str(test_environment["group1_file"]) in handler.filepath_or_buffer
    assert str(test_environment["group2_file"]) in handler.filepath_or_buffer


def test_init_with_single_file(test_environment):
    """Test initialization with a single file path."""
    handler = GroupInteractionHandler(test_environment["group1_file"])
    assert len(handler.filepath_or_buffer) == 1
    assert handler.filepath_or_buffer[0] == str(test_environment["group1_file"])


def test_get_group_filepath(test_environment):
    """Test retrieving a group file path."""
    handler = test_environment["handler"]
    path = handler._get_group_filepath("group1")
    assert path == str(test_environment["group1_file"].resolve())

    with pytest.raises(ValueError) as exc_info:
        handler._get_group_filepath("nonexistent")
    assert (
        str(exc_info.value)
        == "Error: No file found containing 'nonexistent' in its name."
    )


def test_get_group_filepath_non_existent_file(test_environment):
    """Test _get_group_filepath with a non-existent file."""
    non_existent_file = test_environment["test_dir"] / "missing.txt"
    handler = GroupInteractionHandler([non_existent_file])
    with pytest.raises(ValueError) as exc_info:
        handler._get_group_filepath("missing")
    assert (
        str(exc_info.value)
        == f"Error: File does not exist: {non_existent_file.resolve()}"
    )


def test_read_groups(test_environment):
    """Test reading groups from a file."""
    handler = test_environment["handler"]
    groups = handler.read_groups("group1")
    assert len(groups) == 3
    assert groups == ["1_2_3", "4_5_6", "7_8_9"]

    with pytest.raises(ValueError) as exc_info:
        handler.read_groups("")
    assert str(exc_info.value) == "Groups path not specified in configuration"


def test_parse_group_members(test_environment):
    """Test parsing group members from a group ID string."""
    handler = test_environment["handler"]
    members = handler.parse_group_members("1_2_3")
    assert members == [1, 2, 3]

    members = handler.parse_group_members("  4_5_6  ")
    assert members == [4, 5, 6]


def test_get_group_members(test_environment):
    """Test getting group members from string or list."""
    handler = test_environment["handler"]
    # Test with string
    members = handler.get_group_members("1_2_3")
    assert members == [1, 2, 3]

    # Test with list
    members = handler.get_group_members([1, 2, 3])
    assert members == [1, 2, 3]

    # Test empty string
    members = handler.get_group_members("")
    assert members == []

    # Test invalid string
    with pytest.raises(ValueError):
        handler.get_group_members("1_a_3")

    # Test invalid type
    with pytest.raises(TypeError):
        handler.get_group_members(123)


@patch("pygrex.data_reader.DataReader", spec=DataReader)
def test_create_modified_dataset(mock_data_reader, test_environment):
    """Test creating a modified dataset."""
    mock_reader = mock_data_reader()
    mock_reader.dataset = test_environment["test_data"].copy()
    mock_reader.get_new_user_id.side_effect = lambda x: x
    mock_reader.get_new_item_id.side_effect = lambda x: x

    handler = test_environment["handler"]
    # Test with DataReader as original_data
    result_df = handler.create_modified_dataset(mock_reader, [1, 2], [101, 102])
    assert len(result_df) == 6
    assert not (
        (result_df.userId.isin([1, 2])) & (result_df.itemId.isin([101, 102]))
    ).any()

    # Test with DataFrame and DataReader
    result_df = handler.create_modified_dataset(
        test_environment["test_data"], [1, 2], [101, 102], mock_reader
    )
    assert len(result_df) == 6
    assert not (
        (result_df.userId.isin([1, 2])) & (result_df.itemId.isin([101, 102]))
    ).any()

    # Test invalid input
    with pytest.raises(ValueError):
        handler.create_modified_dataset(
            test_environment["test_data"], [1, 2], [101, 102]
        )


@patch("pygrex.data_reader.DataReader", spec=DataReader)
def test_get_rated_items_by_all_group_members(mock_data_reader, test_environment):
    """Test getting items rated by any group member."""
    mock_reader = mock_data_reader()
    mock_reader.dataset = test_environment["test_data"].copy()
    mock_reader.get_new_user_id.side_effect = lambda x: x
    mock_reader.get_original_item_id.side_effect = lambda x: x

    handler = test_environment["handler"]
    group = [1, 2, 3]
    rated_items = handler.get_rated_items_by_all_group_members(group, mock_reader)
    expected_items = np.array([101, 102, 103, 104])
    np.testing.assert_array_equal(np.sort(rated_items), np.sort(expected_items))


@patch("pygrex.data_reader.DataReader", spec=DataReader)
def test_get_common_rated_items(mock_data_reader, test_environment):
    """Test getting items rated by all group members."""
    mock_reader = mock_data_reader()
    mock_reader.dataset = test_environment["test_data"].copy()
    mock_reader.get_new_user_id.side_effect = lambda x: x
    mock_reader.get_original_item_id.side_effect = lambda x: x

    handler = test_environment["handler"]
    # Test group with common item
    group = [1, 2]
    common_items = handler.get_common_rated_items(group, mock_reader)
    np.testing.assert_array_equal(common_items, np.array([102]))

    # Test group with no common items
    group = [1, 4]
    common_items = handler.get_common_rated_items(group, mock_reader)
    assert len(common_items) == 0

    # Test empty group
    group = []
    common_items = handler.get_common_rated_items(group, mock_reader)
    assert len(common_items) == 0
    np.testing.assert_array_equal(common_items, np.array([]))


def test_get_items_for_group_recommendation(test_environment):
    """Test getting items for group recommendation."""
    handler = test_environment["handler"]
    data = pd.DataFrame(
        {"userId": [1, 1, 2, 2, 3], "itemId": [101, 102, 103, 104, 105]}
    )
    all_items = np.array([101, 102, 103, 104, 105, 106, 107])
    group = [1, 2]

    result = handler.get_items_for_group_recommendation(data, all_items, group)
    expected = np.array([105, 106, 107])
    np.testing.assert_array_equal(np.sort(result), np.sort(expected))


@patch("pygrex.data_reader.DataReader", spec=DataReader)
def test_get_group_preferences(mock_data_reader, test_environment):
    """Test getting preferences for group members."""
    mock_reader = mock_data_reader()
    mock_reader.dataset = test_environment["test_data"].copy()
    mock_reader.get_new_user_id.side_effect = lambda x: x

    handler = test_environment["handler"]
    group = [1, 2]
    prefs = handler.get_group_preferences(group, mock_reader)
    assert len(prefs) == 4
    assert prefs.userId.isin([1, 2]).all()


@patch("pygrex.data_reader.DataReader", spec=DataReader)
def test_get_group_preferences_edge_cases(mock_data_reader, test_environment):
    """Test get_group_preferences with string IDs and empty group."""
    mock_reader = mock_data_reader()
    mock_reader.dataset = test_environment["test_data"].copy()
    mock_reader.get_new_user_id.side_effect = (
        lambda x: int(x) if isinstance(x, str) else x
    )

    handler = test_environment["handler"]
    # Test with string IDs
    group = ["1", "2"]
    prefs = handler.get_group_preferences(group, mock_reader)
    assert len(prefs) == 4
    assert prefs.userId.isin([1, 2]).all()

    # Test with empty group
    group = []
    prefs = handler.get_group_preferences(group, mock_reader)
    assert len(prefs) == 0
    assert prefs.empty
