import itertools
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Union

from pygrex.data_reader.data_reader import DataReader
from pygrex.models.recommender_model import RecommenderModel
from pygrex.recommender.group_recommender import GroupRecommender
from pygrex.utils.aggregation_strategy import AggregationStrategy, ScoreAggregator
from pygrex.utils.scale import Scale


class TestGroupRecommender:
    """Test suite for the GroupRecommender class."""

    @pytest.fixture
    def mock_data_reader(self):
        """Create a mock DataReader."""
        mock_data = MagicMock(spec=DataReader)
        # Setup mock dataset
        mock_data.dataset = MagicMock()
        return mock_data

    @pytest.fixture
    def mock_model(self):
        """Create a mock RecommenderModel."""
        return MagicMock(spec=RecommenderModel)

    @pytest.fixture
    def group_recommender(self, mock_data_reader):
        """Create a GroupRecommender instance with mock data."""
        return GroupRecommender(mock_data_reader)

    def test_init(self, mock_data_reader):
        """Test the initialization of GroupRecommender."""
        recommender = GroupRecommender(mock_data_reader)

        assert recommender.data == mock_data_reader
        assert recommender._group_predictions is None
        assert recommender._members is None
        assert recommender._item_pool is None
        assert recommender._model is None
        assert recommender._aggregation_strategy is None
        assert recommender._score_aggregator is None
        assert recommender._aggregated_scores is None

    def test_setup_recommendation(
        self, group_recommender, mock_model, mock_data_reader
    ):
        """Test setup_recommendation method."""
        # Arrange
        members = [1, 2, 3]
        item_ids = [101, 102, 103, 104, 105]
        mock_item_pool = np.array([101, 103, 105])
        aggregation_strategy = AggregationStrategy.AVG_PREDICTIONS
        mock_data_reader.dataset.__getitem__().unique.return_value = item_ids

        # Mock methods
        group_recommender.get_non_interacted_items_for_recommendation = MagicMock(
            return_value=mock_item_pool
        )
        group_recommender._generate_group_predictions = MagicMock(
            return_value={1: {101: 4.5}, 2: {103: 3.2}, 3: {105: 4.0}}
        )
        group_recommender._aggregate_group_scores = MagicMock(
            return_value={101: 4.5, 103: 3.2, 105: 4.0}
        )

        # Act
        group_recommender.setup_recommendation(
            mock_model, members, mock_data_reader, aggregation_strategy
        )

        # Assert
        assert group_recommender._members == members
        assert group_recommender._model == mock_model
        assert group_recommender._aggregation_strategy == aggregation_strategy
        assert isinstance(group_recommender._score_aggregator, ScoreAggregator)
        assert np.array_equal(group_recommender._item_pool, mock_item_pool)
        group_recommender.get_non_interacted_items_for_recommendation.assert_called_once_with(
            group_recommender.data, item_ids, members
        )
        group_recommender._generate_group_predictions.assert_called_once()
        group_recommender._aggregate_group_scores.assert_called_once()

    def test_setup_recommendation_with_most_respected_person(
        self, group_recommender, mock_model, mock_data_reader
    ):
        """Test setup_recommendation method with most respected person."""
        # Arrange
        members = [1, 2, 3]
        item_ids = [101, 102, 103, 104, 105]
        mock_item_pool = np.array([101, 103, 105])
        aggregation_strategy = AggregationStrategy.MOST_RESPECTED_PERSON
        most_respected_person = 1
        mock_data_reader.dataset.__getitem__().unique.return_value = item_ids

        # Mock methods
        group_recommender.get_non_interacted_items_for_recommendation = MagicMock(
            return_value=mock_item_pool
        )
        group_recommender._generate_group_predictions = MagicMock(
            return_value={1: {101: 4.5}, 2: {103: 3.2}, 3: {105: 4.0}}
        )
        group_recommender._aggregate_group_scores = MagicMock(
            return_value={101: 4.5, 103: 3.2, 105: 4.0}
        )

        # Act
        group_recommender.setup_recommendation(
            mock_model,
            members,
            mock_data_reader,
            aggregation_strategy,
            most_respected_person,
        )

        # Assert
        assert (
            group_recommender._score_aggregator.most_respected_person
            == most_respected_person
        )

    def test_generate_group_predictions(self, group_recommender, mock_model):
        """Test _generate_group_predictions method."""
        # Arrange
        members = [1, 2]
        item_pool = np.array([101, 103])
        group_recommender._members = members
        group_recommender._model = mock_model
        group_recommender._item_pool = item_pool

        # Mock generate_recommendation to return different predictions for each user
        group_recommender.generate_recommendation = MagicMock(
            side_effect=[
                {101: 4.5, 103: 3.8},  # User 1's predictions
                {101: 3.2, 103: 4.7},  # User 2's predictions
            ]
        )

        # Act
        result = group_recommender._generate_group_predictions()

        # Assert
        expected = {1: {101: 4.5, 103: 3.8}, 2: {101: 3.2, 103: 4.7}}
        assert result == expected
        assert group_recommender.generate_recommendation.call_count == 2

    def test_generate_group_predictions_error(self, group_recommender):
        """Test _generate_group_predictions method raises error when setup is incomplete."""
        # Arrange - incomplete setup
        group_recommender._members = [1, 2]
        group_recommender._model = None  # Missing model
        group_recommender._item_pool = np.array([101, 103])

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="You must call setup_recommendation before generating predictions",
        ):
            group_recommender._generate_group_predictions()

    def test_aggregate_group_scores(self, group_recommender):
        """Test _aggregate_group_scores method."""
        # Arrange
        group_recommender._group_predictions = {
            1: {101: 4.5, 102: 3.8},
            2: {101: 3.2, 102: 4.7},
        }
        group_recommender._aggregation_strategy = AggregationStrategy.AVG_PREDICTIONS
        mock_score_aggregator = MagicMock(spec=ScoreAggregator)
        mock_score_aggregator.aggregate_scores.return_value = {102: 4.25, 101: 3.85}
        group_recommender._score_aggregator = mock_score_aggregator

        # Act
        result = group_recommender._aggregate_group_scores()

        # Assert
        expected = {102: 4.25, 101: 3.85}  # Already sorted by score descending
        assert result == expected
        mock_score_aggregator.aggregate_scores.assert_called_once_with(
            evaluations=group_recommender._group_predictions,
            strategy=AggregationStrategy.AVG_PREDICTIONS,
            rankings=None,
        )

    def test_aggregate_group_scores_borda_count(self, group_recommender):
        """Test _aggregate_group_scores method with Borda Count strategy."""
        # Arrange
        group_recommender._group_predictions = {
            1: {101: 4.5, 102: 3.8},
            2: {101: 3.2, 102: 4.7},
        }
        group_recommender._aggregation_strategy = AggregationStrategy.BORDA_COUNT
        mock_score_aggregator = MagicMock(spec=ScoreAggregator)
        mock_score_aggregator.aggregate_scores.return_value = {102: 2.0, 101: 1.0}
        group_recommender._score_aggregator = mock_score_aggregator

        # Mock _create_rankings_from_predictions
        mock_rankings = {1: [101, 102], 2: [102, 101]}
        group_recommender._create_rankings_from_predictions = MagicMock(
            return_value=mock_rankings
        )

        # Act
        result = group_recommender._aggregate_group_scores()

        # Assert
        expected = {102: 2.0, 101: 1.0}  # Already sorted by score descending
        assert result == expected
        group_recommender._create_rankings_from_predictions.assert_called_once()
        mock_score_aggregator.aggregate_scores.assert_called_once_with(
            evaluations=group_recommender._group_predictions,
            strategy=AggregationStrategy.BORDA_COUNT,
            rankings=mock_rankings,
        )

    def test_aggregate_group_scores_error(self, group_recommender):
        """Test _aggregate_group_scores method raises error when setup is incomplete."""
        # Arrange - incomplete setup
        group_recommender._group_predictions = None

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="You must call setup_recommendation before aggregating scores",
        ):
            group_recommender._aggregate_group_scores()

    def test_create_rankings_from_predictions(self, group_recommender):
        """Test _create_rankings_from_predictions method."""
        # Arrange
        group_recommender._group_predictions = {
            1: {101: 4.5, 102: 3.8, 103: 2.1},
            2: {101: 3.2, 102: 4.7, 103: 3.9},
        }

        # Act
        result = group_recommender._create_rankings_from_predictions()

        # Assert
        expected = {
            1: [101, 102, 103],  # Sorted by score descending: 4.5, 3.8, 2.1
            2: [102, 103, 101],  # Sorted by score descending: 4.7, 3.9, 3.2
        }
        assert result == expected

    def test_create_rankings_from_predictions_error(self, group_recommender):
        """Test _create_rankings_from_predictions method raises error when predictions are not available."""
        # Arrange - no predictions
        group_recommender._group_predictions = None

        # Act & Assert
        with pytest.raises(ValueError, match="Group predictions not available"):
            group_recommender._create_rankings_from_predictions()

    def test_get_non_interacted_items(self, group_recommender, mock_data_reader):
        """Test get_non_interacted_items_for_recommendation method."""
        # Arrange
        members = [1, 2]
        all_items = [101, 102, 103, 104, 105]

        # Setup mock data
        interacted_items = np.array([102, 104])
        mock_data_reader.dataset.loc = MagicMock()
        mock_data_reader.dataset.loc.__getitem__.return_value.unique.return_value = (
            interacted_items
        )

        # Act
        with patch(
            "numpy.setdiff1d", return_value=np.array([101, 103, 105])
        ) as mock_setdiff:
            result = group_recommender.get_non_interacted_items_for_recommendation(
                mock_data_reader, all_items, members
            )

        # Assert
        assert np.array_equal(result, np.array([101, 103, 105]))
        mock_setdiff.assert_called_once_with(
            all_items, interacted_items, assume_unique=True
        )

    def test_generate_recommendation(
        self, group_recommender, mock_data_reader, mock_model
    ):
        """Test generate_recommendation method."""
        # Arrange
        member = "1"  # Test string conversion
        member_id_int = 1
        new_member_id = 101  # Mapped internal ID
        item_pool = [201, 202]

        # Setup mocks
        mock_data_reader.get_new_user_id.return_value = new_member_id
        mock_data_reader.get_original_item_id.side_effect = (
            lambda x: x + 1000
        )  # Simple mapping function

        mock_model.predict.side_effect = itertools.cycle(
            [3.5, 4.2]
        )  # Predictions for the two items

        # Mock Scale.linear
        with patch(
            "pygrex.utils.scale.Scale.linear", return_value=np.array([3.0, 4.0])
        ) as mock_scale:
            # Act
            result = group_recommender.generate_recommendation(
                mock_model, member, item_pool, mock_data_reader
            )

        # Assert
        mock_data_reader.get_new_user_id.assert_called_once_with(member_id_int)
        assert mock_model.predict.call_count == 2
        mock_scale.assert_called_once()

        # Check if the result dict has the expected structure: {original_item_id: scaled_score}
        expected = {1202: 4.0, 1201: 3.0}  # Sorted by score descending
        assert result == expected

    def test_get_group_recommendations_all(self, group_recommender):
        """Test get_group_recommendations method for returning all items."""
        # Arrange
        group_recommender._aggregated_scores = {102: 4.25, 101: 3.85}

        # Act
        result = group_recommender.get_group_recommendations()

        # Assert
        expected = [102, 101]  # All item IDs from the aggregated scores
        assert result == expected

    def test_get_group_recommendations_top_k(self, group_recommender):
        """Test get_group_recommendations method for returning top k items."""
        # Arrange
        group_recommender._aggregated_scores = {102: 4.25, 101: 3.85, 103: 3.2}

        # Act
        result = group_recommender.get_group_recommendations(top_k=2)

        # Assert
        expected = [102, 101]  # Top 2 items from the aggregated scores
        assert result == expected

    def test_get_group_recommendations_top_one(self, group_recommender):
        """Test get_group_recommendations method for returning only the top item."""
        # Arrange
        group_recommender._aggregated_scores = {102: 4.25, 101: 3.85}

        # Act
        result = group_recommender.get_group_recommendations(top_k=1)

        # Assert
        expected = 102  # The top item ID
        assert result == expected

    def test_get_group_recommendations_error(self, group_recommender):
        """Test get_group_recommendations method raises error when setup is incomplete."""
        # Arrange - incomplete setup
        group_recommender._aggregated_scores = None

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="You must call setup_recommendation before getting recommendations",
        ):
            group_recommender.get_group_recommendations()

    def test_get_top_recommendation(self, group_recommender):
        """Test get_top_recommendation method."""
        # Arrange
        top_item = 102
        group_recommender.get_group_recommendations = MagicMock(return_value=top_item)

        # Act
        result = group_recommender.get_top_recommendation()

        # Assert
        assert result == top_item
        group_recommender.get_group_recommendations.assert_called_once_with(top_k=1)

    def test_get_recommendation_scores(self, group_recommender):
        """Test get_recommendation_scores method."""
        # Arrange
        group_recommender._aggregated_scores = {103: 3.5, 101: 3.5, 102: 3.5}

        # Act
        result = group_recommender.get_recommendation_scores()

        # Assert
        expected = {103: 3.5, 101: 3.5, 102: 3.5}
        assert result == expected

    def test_get_recommendation_scores_error(self, group_recommender):
        """Test get_recommendation_scores method raises error when setup is incomplete."""
        # Arrange - incomplete setup
        group_recommender._aggregated_scores = None

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="You must call setup_recommendation before getting recommendation scores",
        ):
            group_recommender.get_recommendation_scores()

    def test_get_aggregation_strategy(self, group_recommender):
        """Test get_aggregation_strategy method."""
        # Arrange
        strategy = AggregationStrategy.LEAST_MISERY
        group_recommender._aggregation_strategy = strategy

        # Act
        result = group_recommender.get_aggregation_strategy()

        # Assert
        assert result == strategy

    def test_get_group_members(self, group_recommender):
        """Test get_group_members method."""
        # Arrange
        members = [1, 2, 3]
        group_recommender._members = members

        # Act
        result = group_recommender.get_group_members()

        # Assert
        assert result == members
        # Ensure it returns a copy, not the original
        assert result is not members

    def test_get_individual_predictions(self, group_recommender):
        """Test get_individual_predictions method."""
        # Arrange
        predictions = {1: {101: 4.5, 102: 3.8}, 2: {101: 3.2, 102: 4.7}}
        group_recommender._group_predictions = predictions

        # Act
        result = group_recommender.get_individual_predictions()

        # Assert
        assert result == predictions
        # Ensure it returns a copy, not the original
        assert result is not predictions

    def test_get_individual_predictions_none(self, group_recommender):
        """Test get_individual_predictions method when predictions are None."""
        # Arrange
        group_recommender._group_predictions = None

        # Act
        result = group_recommender.get_individual_predictions()

        # Assert
        assert result is None
