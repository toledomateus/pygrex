import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Import the class to be tested
from pygrex.data_reader.data_reader import DataReader
from pygrex.evaluator.sliding_window_evaluator import SlidingWindowEvaluator


@pytest.fixture
def mock_data_reader():
    """Create a mock DataReader instance for testing."""
    mock_reader = MagicMock(spec=DataReader)

    # Sample dataset with user-item interactions
    dataset = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4],
            "itemId": [101, 102, 101, 103, 102, 104, 105],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5, 3.0, 3.8],
        }
    )

    # Set up the mock to return the sample dataset
    mock_reader.dataset = dataset

    # Set up ID mapping methods
    mock_reader.get_new_user_id = lambda user_id: user_id
    mock_reader.get_new_item_id = lambda item_id: item_id

    return mock_reader


@pytest.fixture
def evaluator():
    """Create a SlidingWindowEvaluator instance for testing."""
    config = {"test_param": "test_value"}
    return SlidingWindowEvaluator(config)


@pytest.fixture
def group_predictions():
    """Sample prediction data for multiple users."""
    return {
        1: {101: 4.2, 102: 3.7, 103: 2.5, 104: 1.8},
        2: {101: 4.8, 102: 3.1, 103: 2.2},
        3: {101: 3.9, 102: 4.3, 104: 3.2},
        4: {101: 3.5, 105: 4.0},
    }


class TestSlidingWindowEvaluator:
    def test_initialization(self, evaluator):
        """Test that the evaluator initializes with the correct configuration."""
        assert evaluator.config == {"test_param": "test_value"}
        assert evaluator.group_predictions is None
        assert evaluator.top_recommendation is None

    def test_set_group_recommender_values(self, evaluator, group_predictions):
        """Test setting group recommender values."""
        evaluator.set_group_recommender_values(group_predictions, 101)

        assert evaluator.group_predictions == group_predictions
        assert evaluator.top_recommendation == 101

    def test_calculate_item_popularity_score(self, evaluator, mock_data_reader):
        """Test calculating item popularity scores."""
        items = [101, 102, 103, 104, 105]

        # Expected counts based on mock dataset:
        # item 101: 2 interactions, item 102: 2 interactions, item 103: 1 interaction,
        # item 104: 1 interaction, item 105: 1 interaction
        popularity_scores = evaluator.calculate_item_popularity_score(
            items, mock_data_reader
        )

        # Verify that more popular items have higher scores
        assert (
            popularity_scores[101] == popularity_scores[102]
        )  # Both have 2 interactions
        assert (
            popularity_scores[101] > popularity_scores[103]
        )  # 2 interactions > 1 interaction

        # Check that all items have scores between 0 and 1
        for item_id, score in popularity_scores.items():
            assert 0 <= score <= 1

    def test_calculate_relevance_mask_with_predictions(
        self, evaluator, group_predictions
    ):
        """Test calculating relevance mask when predictions are available."""
        evaluator.set_group_recommender_values(group_predictions, 101)

        # Test for item that all users have predictions for
        relevance_mask = evaluator.calculate_relevance_mask(101)
        assert relevance_mask == {1: 4.2, 2: 4.8, 3: 3.9, 4: 3.5}

        # Test for item that some users don't have predictions for
        relevance_mask = evaluator.calculate_relevance_mask(105)
        assert relevance_mask == {1: 0, 2: 0, 3: 0, 4: 4.0}

        # Test for item no user has predictions for
        relevance_mask = evaluator.calculate_relevance_mask(999)
        assert relevance_mask == {1: 0, 2: 0, 3: 0, 4: 0}

    def test_calculate_relevance_mask_without_predictions(self, evaluator):
        """Test calculating relevance mask when predictions are not set."""
        with pytest.raises(ValueError, match="User predictions not set"):
            evaluator.calculate_relevance_mask(101)

    def test_calculate_relevance_score(
        self, evaluator, mock_data_reader, group_predictions
    ):
        """Test calculating relevance score for an item."""
        evaluator.set_group_recommender_values(group_predictions, 101)
        prediction_scores = {1: 4.2, 2: 4.8, 3: 3.9, 4: 3.5}
        members = [1, 2, 3, 4]

        # Test for item with good data
        relevance_score = evaluator.calculate_relevance_score(
            101, mock_data_reader, prediction_scores, members
        )
        assert 0 <= relevance_score <= 1

        # Test with empty members list
        relevance_score = evaluator.calculate_relevance_score(
            101, mock_data_reader, prediction_scores, []
        )
        assert relevance_score == 0

        # Test with no valid users (no one has interacted with the item)
        relevance_score = evaluator.calculate_relevance_score(
            999, mock_data_reader, prediction_scores, members
        )
        assert relevance_score == 0

    def test_calculate_item_intensity_score(self, evaluator, mock_data_reader):
        """Test calculating item intensity scores."""
        # Test with normal group
        members = [1, 2, 3, 4]

        # Item 101 has been interacted with by users 1 and 2 (2/4 = 0.5)
        intensity = evaluator.calculate_item_intensity_score(
            101, members, mock_data_reader
        )
        assert intensity == 0.5

        # Item 102 has been interacted with by users 1 and 3 (2/4 = 0.5)
        intensity = evaluator.calculate_item_intensity_score(
            102, members, mock_data_reader
        )
        assert intensity == 0.5

        # Item 105 has been interacted with by user 4 only (1/4 = 0.25)
        intensity = evaluator.calculate_item_intensity_score(
            105, members, mock_data_reader
        )
        assert intensity == 0.25

        # Test with empty members list
        intensity = evaluator.calculate_item_intensity_score(101, [], mock_data_reader)
        assert intensity == 0

    def test_calculate_rating_score(self, evaluator, mock_data_reader):
        """Test calculating rating scores."""
        members = [1, 2, 3, 4]

        # Item 101 has ratings from users 1 (4.0) and 2 (5.0)
        # Average over all members: (4.0 + 5.0) / 4 = 2.25
        rating_score = evaluator.calculate_rating_score(101, members, mock_data_reader)
        assert 0 <= rating_score <= 1

        # Test with empty members list
        rating_score = evaluator.calculate_rating_score(101, [], mock_data_reader)
        assert rating_score == 0

    def test_generate_ranked_items(
        self, evaluator, mock_data_reader, group_predictions
    ):
        """Test generating ranked items based on various scores."""
        evaluator.set_group_recommender_values(group_predictions, 101)
        all_rated_items = [101, 102, 103, 104, 105]
        members = [1, 2, 3, 4]

        # Test with default weights
        ranked_items = evaluator.generate_ranked_items(
            all_rated_items, mock_data_reader, members
        )
        assert isinstance(ranked_items, list)
        assert len(ranked_items) == len(all_rated_items)
        assert set(ranked_items) == set(all_rated_items)

        # Test with custom weights
        custom_weights = {
            "popularity": 2.0,
            "intensity": 0.5,
            "rating": 1.0,
            "relevance": 1.5,
            "trend": 0.0,
        }
        ranked_items_custom = evaluator.generate_ranked_items(
            all_rated_items, mock_data_reader, members, custom_weights
        )
        assert isinstance(ranked_items_custom, list)
        assert len(ranked_items_custom) == len(all_rated_items)

        # Test without group predictions set
        evaluator.group_predictions = None
        with pytest.raises(ValueError, match="User predictions not set"):
            evaluator.generate_ranked_items(all_rated_items, mock_data_reader, members)

    def test_evaluate_not_implemented(self, evaluator, mock_data_reader):
        """Test that the evaluate method is defined but not implemented."""
        # The evaluate method should be defined but returns None (pass)
        result = evaluator.evaluate(mock_data_reader)
        assert result is None


# Additional tests for edge cases


def test_with_numpy_user_ids(evaluator, mock_data_reader):
    """Test handling of numpy integer user IDs."""
    # Set up a group with numpy integer user IDs
    np_members = [np.int64(1), np.int64(2), np.int64(3)]

    # Should not raise an error and handle numpy integers correctly
    intensity = evaluator.calculate_item_intensity_score(
        101, np_members, mock_data_reader
    )
    assert 0 <= intensity <= 1

    rating_score = evaluator.calculate_rating_score(101, np_members, mock_data_reader)
    assert 0 <= rating_score <= 1


def test_with_different_rating_scale(evaluator, mock_data_reader):
    """Test using a different rating scale for normalization."""
    members = [1, 2, 3, 4]
    custom_scale = (1, 10)  # 1-10 rating scale

    # Calculate score with custom rating scale
    rating_score = evaluator.calculate_rating_score(
        101, members, mock_data_reader, rating_scale=custom_scale
    )
    assert 0 <= rating_score <= 1

    relevance_score = evaluator.calculate_relevance_score(
        101, mock_data_reader, {1: 8, 2: 9}, members, rating_scale=custom_scale
    )
    assert 0 <= relevance_score <= 1
