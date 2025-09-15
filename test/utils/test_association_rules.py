import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from pygrex.data_reader.data_reader import DataReader
from pygrex.utils.association_rules import AssociationRules


class TestAssociationRulesInitialization:
    """Test class for AssociationRules initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock DataReader with sample data
        self.mock_data_reader = Mock(spec=DataReader)
        self.sample_dataset = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
                "itemId": ["A", "B", "C", "A", "B", "B", "C", "D", "A", "D"],
                "rating": [4.5, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0, 4.5, 5.0, 3.0],
            }
        )
        self.mock_data_reader.dataset = self.sample_dataset

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        ar = AssociationRules(self.mock_data_reader)

        assert ar.data == self.mock_data_reader
        assert ar.min_support == 0.2
        assert ar.min_confidence == 0.2
        assert ar.rating_threshold == 4.0
        assert ar._frequent_itemsets is None
        assert ar._association_rules is None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        ar = AssociationRules(
            self.mock_data_reader,
            min_support=0.1,
            min_confidence=0.3,
            rating_threshold=3.5,
        )

        assert ar.min_support == 0.1
        assert ar.min_confidence == 0.3
        assert ar.rating_threshold == 3.5

    def test_init_with_invalid_min_support(self):
        """Test initialization with invalid min_support values."""
        with pytest.raises(ValueError, match="min_support must be between 0 and 1"):
            AssociationRules(self.mock_data_reader, min_support=0)

        with pytest.raises(ValueError, match="min_support must be between 0 and 1"):
            AssociationRules(self.mock_data_reader, min_support=1.5)

        with pytest.raises(ValueError, match="min_support must be between 0 and 1"):
            AssociationRules(self.mock_data_reader, min_support=-0.1)

    def test_init_with_invalid_min_confidence(self):
        """Test initialization with invalid min_confidence values."""
        with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
            AssociationRules(self.mock_data_reader, min_confidence=0)

        with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
            AssociationRules(self.mock_data_reader, min_confidence=2.0)

    def test_init_with_invalid_rating_threshold(self):
        """Test initialization with invalid rating_threshold."""
        with pytest.raises(ValueError, match="rating_threshold must be non-negative"):
            AssociationRules(self.mock_data_reader, rating_threshold=-1.0)


class TestAssociationRulesValidation:
    """Test class for parameter validation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)
        self.mock_data_reader.dataset = pd.DataFrame(
            {"userId": [1, 2], "itemId": ["A", "B"], "rating": [4.0, 5.0]}
        )

    def test_validate_parameters_valid_inputs(self):
        """Test parameter validation with valid inputs."""
        ar = AssociationRules(self.mock_data_reader)
        # Should not raise any exception
        ar._validate_parameters(0.1, 0.2, 3.0)

    def test_validate_parameters_invalid_support(self):
        """Test parameter validation with invalid support values."""
        ar = AssociationRules(self.mock_data_reader)

        with pytest.raises(ValueError):
            ar._validate_parameters(0, 0.5, 3.0)

        with pytest.raises(ValueError):
            ar._validate_parameters(1.1, 0.5, 3.0)

    def test_validate_parameters_invalid_confidence(self):
        """Test parameter validation with invalid confidence values."""
        ar = AssociationRules(self.mock_data_reader)

        with pytest.raises(ValueError):
            ar._validate_parameters(0.1, 0, 3.0)

        with pytest.raises(ValueError):
            ar._validate_parameters(0.1, 1.5, 3.0)

    def test_validate_parameters_invalid_rating_threshold(self):
        """Test parameter validation with invalid rating threshold."""
        ar = AssociationRules(self.mock_data_reader)

        with pytest.raises(ValueError):
            ar._validate_parameters(0.1, 0.5, -1.0)


class TestAssociationRulesTransactionPreparation:
    """Test class for transaction preparation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)

    def test_prepare_transactions_normal_case(self):
        """Test transaction preparation with normal data."""
        dataset = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 3, 3],
                "itemId": ["A", "B", "C", "A", "B", "B", "C"],
                "rating": [4.5, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            }
        )
        self.mock_data_reader.dataset = dataset

        ar = AssociationRules(self.mock_data_reader, rating_threshold=4.0)
        transactions = ar._prepare_transactions()

        expected_transactions = [
            ["A", "C"],  # User 1: ratings 4.5, 5.0
            ["A", "B"],  # User 2: ratings 4.0, 4.5
            ["C"],  # User 3: rating 4.0
        ]

        assert len(transactions) == 3
        assert transactions == expected_transactions

    def test_prepare_transactions_empty_after_filter(self):
        """Test transaction preparation when no ratings meet threshold."""
        dataset = pd.DataFrame(
            {
                "userId": [1, 2, 3],
                "itemId": ["A", "B", "C"],
                "rating": [2.0, 3.0, 3.5],
            }
        )
        self.mock_data_reader.dataset = dataset

        ar = AssociationRules(self.mock_data_reader, rating_threshold=4.0)

        with pytest.raises(
            ValueError, match="No interactions found with rating >= 4.0"
        ):
            ar._prepare_transactions()

    def test_prepare_transactions_string_conversion(self):
        """Test that movie IDs are converted to strings."""
        dataset = pd.DataFrame(
            {
                "userId": [1, 1],
                "itemId": [123, 456],  # Numeric movie IDs
                "rating": [4.0, 5.0],
            }
        )
        self.mock_data_reader.dataset = dataset

        ar = AssociationRules(self.mock_data_reader)
        transactions = ar._prepare_transactions()

        assert transactions == [["123", "456"]]
        assert all(
            isinstance(item, str)
            for transaction in transactions
            for item in transaction
        )


class TestAssociationRulesMining:
    """Test class for frequent itemsets mining methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)

    @patch("pygrex.utils.association_rules.fpgrowth")
    @patch("pygrex.utils.association_rules.TransactionEncoder")
    def test_mine_frequent_itemsets_success(self, mock_encoder_class, mock_fpgrowth):
        """Test successful frequent itemsets mining."""
        # Mock TransactionEncoder
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder
        mock_encoder.fit_transform.return_value = np.array(
            [[True, False], [False, True]]
        )
        mock_encoder.columns_ = ["A", "B"]

        # Mock fpgrowth result
        mock_frequent_itemsets = pd.DataFrame(
            {"support": [0.3, 0.4], "itemsets": [{"A"}, {"B"}]}
        )
        mock_fpgrowth.return_value = mock_frequent_itemsets

        dataset = pd.DataFrame(
            {"userId": [1, 2], "itemId": ["A", "B"], "rating": [4.0, 5.0]}
        )
        self.mock_data_reader.dataset = dataset

        ar = AssociationRules(self.mock_data_reader, min_support=0.2)
        transactions = [["A"], ["B"]]

        result = ar._mine_frequent_itemsets(transactions)  # type: ignore

        assert not result.empty
        mock_fpgrowth.assert_called_once()
        mock_encoder.fit_transform.assert_called_once_with(transactions)

    @patch("pygrex.utils.association_rules.fpgrowth")
    @patch("pygrex.utils.association_rules.TransactionEncoder")
    def test_mine_frequent_itemsets_empty_result(
        self, mock_encoder_class, mock_fpgrowth
    ):
        """Test frequent itemsets mining with empty result."""
        # Mock TransactionEncoder
        mock_encoder = MagicMock()
        mock_encoder_class.return_value = mock_encoder
        mock_encoder.fit_transform.return_value = np.array([[True, False]])
        mock_encoder.columns_ = ["A", "B"]

        # Mock empty fpgrowth result
        mock_fpgrowth.return_value = pd.DataFrame()

        dataset = pd.DataFrame({"userId": [1], "itemId": ["A"], "rating": [4.0]})
        self.mock_data_reader.dataset = dataset

        ar = AssociationRules(self.mock_data_reader, min_support=0.9)
        transactions = [["A"]]

        with pytest.raises(
            ValueError, match="No frequent itemsets found with min_support=0.9"
        ):
            ar._mine_frequent_itemsets(transactions)  # type: ignore


class TestAssociationRulesGeneration:
    """Test class for association rules generation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_frequent_itemsets = pd.DataFrame(
            {"support": [0.3, 0.4, 0.2], "itemsets": [{"A"}, {"B"}, {"A", "B"}]}
        )

    @patch("pygrex.utils.association_rules.association_rules")
    def test_generate_association_rules_success(self, mock_association_rules):
        """Test successful association rules generation."""
        mock_rules = pd.DataFrame(
            {
                "antecedents": [{"A"}],
                "consequents": [{"B"}],
                "confidence": [0.8],
                "support": [0.2],
            }
        )
        mock_association_rules.return_value = mock_rules

        mock_data_reader = Mock(spec=DataReader)
        ar = AssociationRules(mock_data_reader, min_confidence=0.5)

        result = ar._generate_association_rules(self.sample_frequent_itemsets)

        assert not result.empty
        mock_association_rules.assert_called_once_with(
            self.sample_frequent_itemsets, metric="confidence", min_threshold=0.5
        )

    @patch("pygrex.utils.association_rules.association_rules")
    def test_generate_association_rules_empty_result(self, mock_association_rules):
        """Test association rules generation with empty result."""
        mock_association_rules.return_value = pd.DataFrame()

        mock_data_reader = Mock(spec=DataReader)
        ar = AssociationRules(mock_data_reader, min_confidence=0.9)

        with pytest.raises(
            ValueError, match="No association rules found with min_confidence=0.9"
        ):
            ar._generate_association_rules(self.sample_frequent_itemsets)


class TestAssociationRulesCompute:
    """Test class for the main compute method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)
        self.sample_dataset = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3],
                "itemId": ["A", "B", "A", "C", "B", "C"],
                "rating": [4.0, 5.0, 4.5, 4.0, 4.5, 5.0],
            }
        )
        self.mock_data_reader.dataset = self.sample_dataset

    def test_compute_empty_dataset(self):
        """Test compute method with empty dataset."""
        self.mock_data_reader.dataset = pd.DataFrame()
        ar = AssociationRules(self.mock_data_reader)

        with pytest.raises(ValueError, match="Dataset is empty"):
            ar.compute()

    @patch.object(AssociationRules, "_generate_association_rules")
    @patch.object(AssociationRules, "_mine_frequent_itemsets")
    @patch.object(AssociationRules, "_prepare_transactions")
    def test_compute_success(self, mock_prepare, mock_mine, mock_generate):
        """Test successful compute execution."""
        # Mock method returns
        mock_prepare.return_value = [["A", "B"], ["A", "C"]]
        mock_frequent_itemsets = pd.DataFrame({"support": [0.3], "itemsets": [{"A"}]})
        mock_mine.return_value = mock_frequent_itemsets
        mock_rules = pd.DataFrame(
            {"antecedents": [{"A"}], "consequents": [{"B"}], "confidence": [0.8]}
        )
        mock_generate.return_value = mock_rules

        ar = AssociationRules(self.mock_data_reader)
        result = ar.compute()

        assert result.equals(mock_rules)
        assert ar._frequent_itemsets.equals(mock_frequent_itemsets)  # type: ignore
        assert ar._association_rules.equals(mock_rules)  # type: ignore

        mock_prepare.assert_called_once()
        mock_mine.assert_called_once_with([["A", "B"], ["A", "C"]])
        mock_generate.assert_called_once_with(mock_frequent_itemsets)

    @patch.object(AssociationRules, "_prepare_transactions")
    def test_compute_no_transactions(self, mock_prepare):
        """Test compute method when no transactions are found."""
        mock_prepare.return_value = []

        ar = AssociationRules(self.mock_data_reader)

        with pytest.raises(ValueError, match="No transactions found after filtering"):
            ar.compute()


class TestAssociationRulesAccessors:
    """Test class for accessor methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)
        self.ar = AssociationRules(self.mock_data_reader)

    def test_get_frequent_itemsets_before_compute(self):
        """Test getting frequent itemsets before compute is called."""
        result = self.ar.get_frequent_itemsets()
        assert result is None

    def test_get_frequent_itemsets_after_compute(self):
        """Test getting frequent itemsets after compute is called."""
        mock_itemsets = pd.DataFrame({"support": [0.3], "itemsets": [{"A"}]})
        self.ar._frequent_itemsets = mock_itemsets

        result = self.ar.get_frequent_itemsets()
        assert result.equals(mock_itemsets)  # type: ignore


class TestAssociationRulesRecommendations:
    """Test class for recommendation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)
        self.ar = AssociationRules(self.mock_data_reader)

        # Mock association rules
        self.mock_rules = pd.DataFrame(
            {
                "antecedents": [
                    frozenset(["A"]),
                    frozenset(["B"]),
                    frozenset(["A", "B"]),
                ],
                "consequents": [frozenset(["B"]), frozenset(["C"]), frozenset(["C"])],
                "confidence": [0.8, 0.6, 0.9],
                "lift": [1.2, 1.1, 1.5],
                "support": [0.4, 0.3, 0.2],
            }
        )

    def test_get_recommendations_before_compute(self):
        """Test getting recommendations before compute is called."""
        with pytest.raises(
            RuntimeError, match="Must call compute\\(\\) before getting recommendations"
        ):
            self.ar.get_recommendations_for_items(["A"])

    def test_get_recommendations_empty_items(self):
        """Test getting recommendations with empty items list."""
        self.ar._association_rules = self.mock_rules

        with pytest.raises(ValueError, match="Items list cannot be empty"):
            self.ar.get_recommendations_for_items([])

    def test_get_recommendations_success(self):
        """Test successful recommendation generation."""
        self.ar._association_rules = self.mock_rules

        result = self.ar.get_recommendations_for_items(["A"], top_k=5)

        assert not result.empty
        assert len(result) <= 5
        assert "confidence" in result.columns
        assert "antecedents" in result.columns
        assert "consequents" in result.columns

    def test_get_recommendations_no_matching_rules(self):
        """Test recommendations when no rules match the items."""
        self.ar._association_rules = self.mock_rules

        result = self.ar.get_recommendations_for_items(["Z"])  # Item not in rules

        assert result.empty

    def test_get_recommendations_top_k_limit(self):
        """Test that recommendations respect top_k limit."""
        # Create more rules than top_k
        extended_rules = pd.concat([self.mock_rules] * 5, ignore_index=True)
        self.ar._association_rules = extended_rules

        result = self.ar.get_recommendations_for_items(["A"], top_k=2)

        assert len(result) <= 2


class TestAssociationRulesStringRepresentations:
    """Test class for string representation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_reader = Mock(spec=DataReader)
        self.mock_data_reader.dataset = pd.DataFrame(
            {"userId": [1], "itemId": ["A"], "rating": [4.0]}
        )

    def test_str_representation(self):
        """Test string representation of AssociationRules object."""
        ar = AssociationRules(
            self.mock_data_reader,
            min_support=0.1,
            min_confidence=0.3,
            rating_threshold=3.5,
        )

        expected = "AssociationRules(min_support=0.1, min_confidence=0.3, rating_threshold=3.5)"
        assert str(ar) == expected

    def test_repr_representation(self):
        """Test repr representation of AssociationRules object."""
        ar = AssociationRules(self.mock_data_reader)

        expected = "AssociationRules(min_support=0.2, min_confidence=0.2, rating_threshold=4.0)"
        assert repr(ar) == expected


class TestAssociationRulesIntegration:
    """Integration tests for AssociationRules class."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.mock_data_reader = Mock(spec=DataReader)

        # Create a more comprehensive dataset for integration testing
        self.integration_dataset = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5],
                "itemId": [
                    "A",
                    "B",
                    "C",
                    "A",
                    "B",
                    "D",
                    "B",
                    "C",
                    "A",
                    "C",
                    "D",
                    "B",
                    "D",
                ],
                "rating": [
                    4.5,
                    4.0,
                    5.0,
                    4.0,
                    4.5,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    4.0,
                    4.5,
                    4.0,
                    4.0,
                ],
            }
        )
        self.mock_data_reader.dataset = self.integration_dataset

    @patch("pygrex.utils.association_rules.association_rules")
    @patch("pygrex.utils.association_rules.fpgrowth")
    def test_full_workflow_integration(self, mock_fpgrowth, mock_association_rules):
        """Test the complete workflow from initialization to recommendations."""
        # Mock fpgrowth result
        mock_frequent_itemsets = pd.DataFrame(
            {
                "support": [0.4, 0.6, 0.3],
                "itemsets": [frozenset(["A"]), frozenset(["B"]), frozenset(["A", "B"])],
            }
        )
        mock_fpgrowth.return_value = mock_frequent_itemsets

        # Mock association rules result
        mock_rules = pd.DataFrame(
            {
                "antecedents": [frozenset(["A"])],
                "consequents": [frozenset(["B"])],
                "confidence": [0.8],
                "lift": [1.2],
                "support": [0.3],
            }
        )
        mock_association_rules.return_value = mock_rules

        # Initialize and compute
        ar = AssociationRules(
            self.mock_data_reader,
            min_support=0.2,
            min_confidence=0.5,
            rating_threshold=4.0,
        )

        # Run compute
        rules = ar.compute()

        # Verify results
        assert not rules.empty
        assert ar.get_frequent_itemsets() is not None

        # Test recommendations
        recommendations = ar.get_recommendations_for_items(["A"])
        assert isinstance(recommendations, pd.DataFrame)


# Pytest configuration and fixtures
@pytest.fixture
def sample_data_reader():
    """Fixture providing a sample DataReader for tests."""
    mock_data_reader = Mock(spec=DataReader)
    mock_data_reader.dataset = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "itemId": ["A", "B", "A", "C", "B", "C"],
            "rating": [4.0, 5.0, 4.5, 4.0, 4.5, 5.0],
        }
    )
    return mock_data_reader


@pytest.fixture
def association_rules_instance(sample_data_reader):
    """Fixture providing an AssociationRules instance for tests."""
    return AssociationRules(sample_data_reader)


# Parametrized tests
@pytest.mark.parametrize(
    "support,confidence,threshold,should_raise",
    [
        (0.1, 0.2, 4.0, False),
        (0.0, 0.2, 4.0, True),
        (1.1, 0.2, 4.0, True),
        (0.1, 0.0, 4.0, True),
        (0.1, 1.5, 4.0, True),
        (0.1, 0.2, -1.0, True),
    ],
)
def test_parameter_validation_parametrized(
    sample_data_reader, support, confidence, threshold, should_raise
):
    """Parametrized test for parameter validation."""
    if should_raise:
        with pytest.raises(ValueError):
            AssociationRules(sample_data_reader, support, confidence, threshold)
    else:
        ar = AssociationRules(sample_data_reader, support, confidence, threshold)
        assert ar.min_support == support
        assert ar.min_confidence == confidence
        assert ar.rating_threshold == threshold
