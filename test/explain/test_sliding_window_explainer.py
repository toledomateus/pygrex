import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Union

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.data_reader.group_interaction_handler import GroupInteractionHandler
from pygrex.explain.sliding_window_explainer import SlidingWindowExplainer
from pygrex.models.recommender_model import RecommenderModel
from pygrex.recommender.group_recommender import GroupRecommender


class TestSlidingWindowExplainer:
    """Test suite for the SlidingWindowExplainer class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        mock_cfg = Mock(spec=cfg)
        return mock_cfg

    @pytest.fixture
    def mock_data_reader(self):
        """Create a mock DataReader with sample data."""
        # Create a sample dataset with user-item interactions
        data = {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3],
            "itemId": [101, 102, 103, 101, 104, 102, 103, 105],
            "rating": [5, 4, 3, 4, 5, 3, 4, 5],
        }
        df = pd.DataFrame(data)

        # Create a mock DataReader with the sample dataset
        mock_reader = Mock(spec=DataReader)
        mock_reader.dataset = df

        # Mock ID mapping methods
        mock_reader.get_new_user_id = lambda x: int(
            x
        )  # Just return the ID as is for testing
        mock_reader.get_new_item_id = lambda x: int(
            x
        )  # Just return the ID as is for testing
        mock_reader.make_consecutive_ids_in_dataset = Mock()
        mock_reader.binarize = Mock()

        return mock_reader

    @pytest.fixture
    def mock_group_handler(self):
        """Create a mock GroupInteractionHandler."""
        mock_handler = Mock(spec=GroupInteractionHandler)

        # Mock create_modified_dataset to return modified DataFrame
        def modified_dataset_func(*args, **kwargs):
            # Create a slightly modified version of the original dataset
            modified_df = kwargs.get("original_data").copy()
            return modified_df

        mock_handler.create_modified_dataset = modified_dataset_func
        return mock_handler

    @pytest.fixture
    def mock_recommender_model(self):
        """Create a mock recommender model."""
        mock_model = Mock(spec=RecommenderModel)
        mock_model.fit = Mock(return_value=None)
        return mock_model

    @pytest.fixture
    def mock_sliding_window(self):
        """Create a mock sliding window."""
        mock_window = Mock()

        # Set up the get_next_window method to return windows in sequence
        mock_window.get_next_window = Mock(
            side_effect=[
                [101, 102],  # First window
                [103, 104],  # Second window
                None,  # End of windows
            ]
        )

        return mock_window

    @pytest.fixture
    def mock_group_recommender(self):
        """Create a mock group recommender."""
        with patch(
            "pygrex.recommender.group_recommender.GroupRecommender", autospec=True
        ) as mock_gr:
            # Configure the mock to return predictable recommendations
            instance = mock_gr.return_value
            instance.setup_recommendation = Mock()

            # Make get_group_recommendations return different values based on input
            def get_recommendations(n):
                # Default recommendation includes target item 200
                return [200, 201, 202]

            instance.get_group_recommendations = Mock(side_effect=get_recommendations)
            yield mock_gr

    @pytest.fixture
    def explainer(
        self,
        mock_config,
        mock_data_reader,
        mock_group_handler,
        mock_recommender_model,
        mock_sliding_window,
    ):
        """Create a SlidingWindowExplainer instance with mocked dependencies."""
        # Create an explainer with test data
        explainer = SlidingWindowExplainer(
            cfg=mock_config,
            data=mock_data_reader,
            group_handler=mock_group_handler,
            members=[1, 2, 3],
            target_item=200,
            candidate_items=[200, 201, 202, 203],
            sliding_window=mock_sliding_window,
            model=mock_recommender_model,
        )
        return explainer

    def test_initialization(self, explainer):
        """Test that the explainer initializes with correct attributes."""
        assert explainer.members == [1, 2, 3]
        assert explainer.target_item == 200
        assert explainer.candidate_items == [200, 201, 202, 203]
        assert explainer.calls == 0
        assert explainer.explanations_found == {}

    def test_set_sliding_window(self, explainer):
        """Test setting the sliding window after initialization."""
        new_window = Mock()
        explainer.set_sliding_window(new_window)
        assert explainer.sliding_window == new_window

    def test_find_explanation_no_sliding_window(self, explainer):
        """Test that find_explanation raises an error when no sliding window is set."""
        explainer.sliding_window = None
        with pytest.raises(ValueError) as excinfo:
            explainer.find_explanation()
        assert "Sliding window has not been set" in str(excinfo.value)

    @patch.object(SlidingWindowExplainer, "_test_window_removal")
    @patch.object(SlidingWindowExplainer, "_find_minimal_subset")
    def test_find_explanation_no_explanations_found(
        self, mock_find_minimal, mock_test_window, explainer
    ):
        """Test behavior when no explanations are found."""
        # Make _test_window_removal always return False (no effect on recommendations)
        mock_test_window.return_value = False

        result = explainer.find_explanation()

        # Check that window was tested but no minimal subset was searched
        assert mock_test_window.call_count > 0
        assert mock_find_minimal.call_count == 0
        assert result == {}  # No explanations found

    @patch.object(SlidingWindowExplainer, "_test_window_removal")
    @patch.object(SlidingWindowExplainer, "_find_minimal_subset")
    def test_find_explanation_found(
        self, mock_find_minimal, mock_test_window, explainer
    ):
        """Test behavior when an explanation is found."""
        # Make second window test return True (has effect on recommendations)
        mock_test_window.side_effect = [False, True]

        explainer.find_explanation()

        # Check that minimal subset was searched for the second window
        assert mock_test_window.call_count == 2
        assert mock_find_minimal.call_count == 1
        # Check that the window passed to _find_minimal_subset is the second window
        assert mock_find_minimal.call_args[0][0] == [103, 104]

    @patch("pygrex.utils.scale.Scale.linear")
    @patch("pygrex.explain.sliding_window_explainer.GroupRecommender")
    def test_get_recommendations_after_removal(
        self,
        mock_group_recommender_cls,  # Corresponds to the inner @patch for GroupRecommender class
        mock_scale_linear_method,  # Corresponds to the outer @patch for Scale.linear method
        explainer,
        mock_data_reader,
    ):
        """Test getting recommendations after removing items."""

        # Create a mock GroupRecommender instance
        mock_recommender_instance = Mock()
        mock_group_recommender_cls.return_value = mock_recommender_instance

        # Mock methods on the GroupRecommender instance
        mock_recommender_instance.get_group_recommendations.return_value = [
            201,
            202,
            203,
        ]
        mock_recommender_instance.setup_recommendation = (
            Mock()
        )  # Mock the method itself

        # Mock the internal methods on the ACTUAL 'explainer' (SlidingWindowExplainer instance)
        # 'explainer' here is now the fixture instance, as intended.
        explainer._create_data_reader_and_prepare = Mock(return_value=mock_data_reader)
        explainer._retrain_model = Mock(
            return_value=Mock()
        )  # Assuming _retrain_model returns a model mock
        explainer.group_handler.create_modified_dataset = Mock(
            return_value=mock_data_reader.dataset
        )

        # Test the method on the actual 'explainer' instance
        result = explainer._get_recommendations_after_removal([101, 102])

        # Verify the result
        # 'result' should now be [201, 202, 203] because it comes from
        # mock_recommender_instance.get_group_recommendations
        assert result == [201, 202, 203]

        # Verify that the GroupRecommender CLASS was instantiated with the correct data reader
        # Inside _get_recommendations_after_removal:
        # data_retrained = self._create_data_reader_and_prepare(...) # returns mock_data_reader
        # group_recommender = GroupRecommender(data_retrained)       # This is the call we're checking
        mock_group_recommender_cls.assert_called_once_with(mock_data_reader)

        # Verify that setup_recommendation was called ON THE INSTANCE
        mock_recommender_instance.setup_recommendation.assert_called_once()

        # Verify that get_group_recommendations was called ON THE INSTANCE with default top_n=10
        mock_recommender_instance.get_group_recommendations.assert_called_once_with(10)

    def test_test_window_removal_target_removed(self, explainer):
        """Test that _test_window_removal returns True when target item is removed from recommendations."""
        # Mock _get_recommendations_after_removal to return recommendations without target item
        explainer._get_recommendations_after_removal = Mock(
            return_value=[201, 202, 203]
        )

        result = explainer._test_window_removal([101, 102], 200)

        assert result is True

    def test_test_window_removal_target_still_present(self, explainer):
        """Test that _test_window_removal returns False when target item remains in recommendations."""
        # Mock _get_recommendations_after_removal to return recommendations with target item
        explainer._get_recommendations_after_removal = Mock(
            return_value=[200, 201, 202]
        )

        result = explainer._test_window_removal([103, 104], 200)

        assert result is False

    @patch.object(SlidingWindowExplainer, "_get_recommendations_after_removal")
    @patch.object(SlidingWindowExplainer, "_record_explanation")
    def test_find_minimal_subset_found(self, mock_record, mock_get_recs, explainer):
        """Test finding a minimal subset that produces a counterfactual explanation."""

        # Configure mock to make only [101] affect recommendations (not include target 200)
        def get_recs_side_effect(items, top_n=10):
            if items == [101]:
                return [201, 202, 203]  # Without target item
            else:
                return [200, 201, 202]  # With target item

        mock_get_recs.side_effect = get_recs_side_effect

        # Call the method
        explainer._find_minimal_subset([101, 102], 200)

        # Verify _record_explanation was called with the minimal subset [101]
        assert mock_record.called
        assert mock_record.call_args[0][0] == [101]

    @patch.object(SlidingWindowExplainer, "_get_recommendations_after_removal")
    @patch.object(SlidingWindowExplainer, "_record_explanation")
    def test_find_minimal_subset_not_found(self, mock_record, mock_get_recs, explainer):
        """Test behavior when no minimal subset is found."""
        # Configure mock so no subset affects recommendations
        mock_get_recs.return_value = [200, 201, 202]  # Always includes target item

        # Call the method
        explainer._find_minimal_subset([101, 102], 200)

        # Verify _record_explanation was not called
        assert not mock_record.called

    @patch.object(SlidingWindowExplainer, "_calculate_item_intensity")
    @patch.object(SlidingWindowExplainer, "_calculate_user_intensity")
    def test_record_explanation(
        self, mock_user_intensity, mock_item_intensity, explainer, capfd
    ):
        """Test recording an explanation."""
        # Configure mocks
        mock_item_intensity.return_value = [0.5, 0.7]
        mock_user_intensity.return_value = [0.3, 0.6, 0.8]

        # Call method
        explainer._record_explanation([101, 102], 200, 201)

        # Check explanation was stored
        assert explainer.explanations_found[explainer.calls] == [101, 102]

        # Check print output
        out, _ = capfd.readouterr()
        assert "If the group had not interacted with these items" in out
        assert "Explanation: [101, 102]" in out

    def test_calculate_average_item_intensity_score(self, mock_data_reader):
        """Test calculation of average item intensity."""
        # Use static method directly
        result = SlidingWindowExplainer._calculate_average_item_intensity_score(
            explanation=[101, 102], members=[1, 2, 3], data=mock_data_reader
        )

        # Expected:
        # - Item 101 has interactions with users 1 and 2 (2/3 = 0.67)
        # - Item 102 has interactions with users 1 and 3 (2/3 = 0.67)
        assert len(result) == 2
        assert result[0] == pytest.approx(2 / 3)
        assert result[1] == pytest.approx(2 / 3)

    def test_calculate_user_intensity_score(self, mock_data_reader):
        """Test calculation of user intensity."""
        # Use static method directly
        result = SlidingWindowExplainer._calculate_user_intensity_score(
            explanation_items=[101, 102, 103], members=[1, 2, 3], data=mock_data_reader
        )

        # Expected:
        # - User 1 interacted with items 101, 102, 103 (3/3 = 1.0)
        # - User 2 interacted with item 101 only (1/3 = 0.33)
        # - User 3 interacted with items 102, 103 (2/3 = 0.67)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1 / 3)
        assert result[2] == pytest.approx(2 / 3)

    def test_create_data_reader_and_prepare(self, explainer, mock_data_reader):
        """Test creating and preparing a new DataReader with modified data."""
        with patch(
            "pygrex.explain.sliding_window_explainer.DataReader"
        ) as mock_reader_class:
            # Set up mock DataReader class
            mock_new_reader = Mock(spec=DataReader)
            mock_reader_class.return_value = mock_new_reader

            # Call method
            result = explainer._create_data_reader_and_prepare(mock_data_reader.dataset)

            # Check DataReader was created and methods were called
            assert mock_reader_class.called
            assert mock_new_reader.make_consecutive_ids_in_dataset.called
            assert mock_new_reader.binarize.called
            assert result == mock_new_reader

    def test_retrain_model(self, explainer, mock_data_reader):
        """Test retraining the model with modified data."""
        model = explainer.model
        result = explainer._retrain_model(mock_data_reader)

        # Check that fit was called and the model was returned
        assert model.fit.called
        assert model.fit.call_args[0][0] == mock_data_reader
        assert result == model

    def test_max_calls_limit(self, explainer):
        """Test that find_explanation respects max_calls limit."""
        # Set a very low max_calls value
        explainer.max_calls = 1

        # Mock necessary methods to isolate test
        explainer._test_window_removal = Mock(return_value=False)

        # Call find_explanation
        result = explainer.find_explanation()

        # Verify only one call was made
        assert explainer.calls == 1
        assert result == {}
