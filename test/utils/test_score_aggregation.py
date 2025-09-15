import unittest
import pytest
import numpy as np
from unittest.mock import patch
from typing import Dict, List
from pygrex.utils.aggregation_strategy import (
    ScoreAggregator,
    AggregationStrategy,  # type: ignore
)
from enum import Enum
from typing import TypeAlias


UserID: TypeAlias = str
ItemID: TypeAlias = str
EvaluationScore: TypeAlias = float
AggregatedScore: TypeAlias = float
UserEvaluations: TypeAlias = Dict[UserID, Dict[ItemID, EvaluationScore]]
UserRankings: TypeAlias = Dict[UserID, List[ItemID]]
AggregatedScores: TypeAlias = Dict[ItemID, AggregatedScore]


class TestScoreAggregatorUnittest(unittest.TestCase):
    """Unit tests using unittest framework."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.aggregator = ScoreAggregator(most_respected_person="user1")

        self.sample_evaluations: UserEvaluations = {
            "user1": {"item_A": 4.5, "item_B": 3.0, "item_C": 5.0},
            "user2": {"item_A": 3.0, "item_B": 4.0, "item_C": 2.0},
            "user3": {"item_A": 4.0, "item_B": 2.0, "item_C": 3.0},
        }

        self.sample_rankings: UserRankings = {
            "user1": ["item_C", "item_A", "item_B"],
            "user2": ["item_B", "item_A", "item_C"],
            "user3": ["item_A", "item_C", "item_B"],
        }

        self.empty_evaluations: UserEvaluations = {}

        self.single_user_evaluations: UserEvaluations = {
            "user1": {"item_A": 3.5, "item_B": 4.0}
        }

    def test_init_with_mrp(self):
        """Test initialization with most respected person."""
        aggregator = ScoreAggregator(most_respected_person="user1")
        self.assertEqual(aggregator.most_respected_person, "user1")

    def test_init_without_mrp(self):
        """Test initialization without most respected person."""
        aggregator = ScoreAggregator()
        self.assertIsNone(aggregator.most_respected_person)

    def test_empty_evaluations(self):
        """Test aggregation with empty evaluations."""
        result = self.aggregator.aggregate_scores(
            self.empty_evaluations,  # type: ignore
            AggregationStrategy.AVG_PREDICTIONS,  # type: ignore
        )
        self.assertEqual(result, {})

    def test_avg_predictions(self):
        """Test average predictions aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.AVG_PREDICTIONS,  # type: ignore
        )

        # Expected: item_A: (4.5+3.0+4.0)/3 = 3.833..., item_B: (3.0+4.0+2.0)/3 = 3.0, item_C: (5.0+2.0+3.0)/3 = 3.333...
        self.assertAlmostEqual(result["item_A"], 3.833333333333333, places=5)
        self.assertAlmostEqual(result["item_B"], 3.0, places=5)
        self.assertAlmostEqual(result["item_C"], 3.333333333333333, places=5)

    def test_least_misery(self):
        """Test least misery aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.LEAST_MISERY,  # type: ignore
        )

        # Expected: item_A: min(4.5, 3.0, 4.0) = 3.0, item_B: min(3.0, 4.0, 2.0) = 2.0, item_C: min(5.0, 2.0, 3.0) = 2.0
        self.assertEqual(result["item_A"], 3.0)
        self.assertEqual(result["item_B"], 2.0)
        self.assertEqual(result["item_C"], 2.0)

    def test_most_pleasure(self):
        """Test most pleasure aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.MOST_PLEASURE,  # type: ignore
        )

        # Expected: item_A: max(4.5, 3.0, 4.0) = 4.5, item_B: max(3.0, 4.0, 2.0) = 4.0, item_C: max(5.0, 2.0, 3.0) = 5.0
        self.assertEqual(result["item_A"], 4.5)
        self.assertEqual(result["item_B"], 4.0)
        self.assertEqual(result["item_C"], 5.0)

    def test_most_respected_person(self):
        """Test most respected person aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.MOST_RESPECTED_PERSON,  # type: ignore
        )

        # Expected: user1's evaluations: item_A: 4.5, item_B: 3.0, item_C: 5.0
        self.assertEqual(result["item_A"], 4.5)
        self.assertEqual(result["item_B"], 3.0)
        self.assertEqual(result["item_C"], 5.0)

    def test_mrp_without_mrp_set(self):
        """Test MRP strategy without setting most respected person."""
        aggregator = ScoreAggregator()  # No MRP set

        with self.assertRaises(ValueError) as context:
            aggregator.aggregate_scores(
                self.sample_evaluations,  # type: ignore
                AggregationStrategy.MOST_RESPECTED_PERSON,  # type: ignore
            )

        self.assertIn("Most respected person not specified", str(context.exception))

    def test_mrp_user_not_in_evaluations(self):
        """Test MRP strategy when MRP user is not in evaluations."""
        aggregator = ScoreAggregator(most_respected_person="nonexistent_user")

        with self.assertRaises(ValueError) as context:
            aggregator.aggregate_scores(
                self.sample_evaluations,  # type: ignore
                AggregationStrategy.MOST_RESPECTED_PERSON,  # type: ignore
            )

        self.assertIn("not found in evaluations", str(context.exception))

    def test_additive_utilitarian(self):
        """Test additive utilitarian aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.ADDITIVE_UTILITARIAN,  # type: ignore
        )

        # Expected: item_A: 4.5+3.0+4.0 = 11.5, item_B: 3.0+4.0+2.0 = 9.0, item_C: 5.0+2.0+3.0 = 10.0
        self.assertEqual(result["item_A"], 11.5)
        self.assertEqual(result["item_B"], 9.0)
        self.assertEqual(result["item_C"], 10.0)

    def test_multiplicative(self):
        """Test multiplicative aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.MULTIPLICATIVE,  # type: ignore
        )

        # Expected: item_A: 4.5*3.0*4.0 = 54.0, item_B: 3.0*4.0*2.0 = 24.0, item_C: 5.0*2.0*3.0 = 30.0
        self.assertEqual(result["item_A"], 54.0)
        self.assertEqual(result["item_B"], 24.0)
        self.assertEqual(result["item_C"], 30.0)

    def test_borda_count(self):
        """Test Borda count aggregation."""
        result = self.aggregator.aggregate_scores(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.BORDA_COUNT,
            self.sample_rankings,  # type: ignore
        )

        # Expected scores based on rankings:
        # item_A: user1(1) + user2(1) + user3(2) = 4
        # item_B: user1(0) + user2(2) + user3(0) = 2
        # item_C: user1(2) + user2(0) + user3(1) = 3
        self.assertEqual(result["item_A"], 4.0)
        self.assertEqual(result["item_B"], 2.0)
        self.assertEqual(result["item_C"], 3.0)

    def test_borda_count_without_rankings(self):
        """Test Borda count without providing rankings."""
        with self.assertRaises(ValueError) as context:
            self.aggregator.aggregate_scores(
                self.sample_evaluations,  # type: ignore
                AggregationStrategy.BORDA_COUNT,  # type: ignore
            )

        self.assertIn("Rankings required for Borda Count", str(context.exception))

    def test_unknown_strategy(self):
        """Test with unknown aggregation strategy."""
        with self.assertRaises(ValueError) as context:
            self.aggregator.aggregate_scores(
                self.sample_evaluations,  # type: ignore
                "invalid_strategy",  # type: ignore
            )

        self.assertIn("Unknown aggregation strategy", str(context.exception))

    def test_get_top_recommendation(self):
        """Test getting top recommendation."""
        top_item = self.aggregator.get_top_recommendation(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.MOST_PLEASURE,  # type: ignore
        )

        # item_C has the highest max value (5.0)
        self.assertEqual(top_item, "item_C")

    def test_get_top_recommendation_with_rankings(self):
        """Test getting top recommendation with Borda count."""
        top_item = self.aggregator.get_top_recommendation(
            self.sample_evaluations,  # type: ignore
            AggregationStrategy.BORDA_COUNT,
            self.sample_rankings,  # type: ignore
        )

        # item_A has the highest Borda count (4.0)
        self.assertEqual(top_item, "item_A")

    def test_single_user_evaluation(self):
        """Test aggregation with single user."""
        result = self.aggregator.aggregate_scores(
            self.single_user_evaluations,  # type: ignore
            AggregationStrategy.AVG_PREDICTIONS,  # type: ignore
        )

        # With single user, average should equal the original values
        self.assertEqual(result["item_A"], 3.5)
        self.assertEqual(result["item_B"], 4.0)

    def test_missing_items_in_evaluations(self):
        """Test with missing items in some user evaluations."""
        incomplete_evaluations: UserEvaluations = {
            "user1": {"item_A": 4.0, "item_B": 3.0},
            "user2": {"item_A": 3.0, "item_C": 2.0},  # Missing item_B
            "user3": {"item_B": 2.0, "item_C": 3.0},  # Missing item_A
        }

        result = self.aggregator.aggregate_scores(
            incomplete_evaluations,  # type: ignore
            AggregationStrategy.AVG_PREDICTIONS,  # type: ignore
        )

        # Should handle missing items gracefully
        self.assertAlmostEqual(result["item_A"], 3.5, places=5)  # (4.0 + 3.0) / 2
        self.assertAlmostEqual(result["item_B"], 2.5, places=5)  # (3.0 + 2.0) / 2
        self.assertAlmostEqual(result["item_C"], 2.5, places=5)  # (2.0 + 3.0) / 2


class TestScoreAggregatorPytest:
    """Unit tests using pytest framework."""

    @pytest.fixture
    def aggregator(self):
        """Fixture for ScoreAggregator instance."""
        return ScoreAggregator(most_respected_person="user1")

    @pytest.fixture
    def sample_evaluations(self):
        """Fixture for sample evaluations."""
        return {
            "user1": {"item_A": 4.5, "item_B": 3.0, "item_C": 5.0},
            "user2": {"item_A": 3.0, "item_B": 4.0, "item_C": 2.0},
            "user3": {"item_A": 4.0, "item_B": 2.0, "item_C": 3.0},
        }

    @pytest.fixture
    def sample_rankings(self):
        """Fixture for sample rankings."""
        return {
            "user1": ["item_C", "item_A", "item_B"],
            "user2": ["item_B", "item_A", "item_C"],
            "user3": ["item_A", "item_C", "item_B"],
        }

    def test_avg_predictions_pytest(self, aggregator, sample_evaluations):
        """Test average predictions using pytest."""
        result = aggregator.aggregate_scores(
            sample_evaluations, AggregationStrategy.AVG_PREDICTIONS
        )

        assert abs(result["item_A"] - 3.833333333333333) < 1e-5
        assert abs(result["item_B"] - 3.0) < 1e-5
        assert abs(result["item_C"] - 3.333333333333333) < 1e-5

    def test_least_misery_pytest(self, aggregator, sample_evaluations):
        """Test least misery using pytest."""
        result = aggregator.aggregate_scores(
            sample_evaluations, AggregationStrategy.LEAST_MISERY
        )

        assert result["item_A"] == 3.0
        assert result["item_B"] == 2.0
        assert result["item_C"] == 2.0

    def test_borda_count_pytest(self, aggregator, sample_evaluations, sample_rankings):
        """Test Borda count using pytest."""
        result = aggregator.aggregate_scores(
            sample_evaluations, AggregationStrategy.BORDA_COUNT, sample_rankings
        )

        assert result["item_A"] == 4.0
        assert result["item_B"] == 2.0
        assert result["item_C"] == 3.0

    def test_mrp_error_pytest(self, sample_evaluations):
        """Test MRP error handling using pytest."""
        aggregator = ScoreAggregator()  # No MRP set

        with pytest.raises(ValueError, match="Most respected person not specified"):
            aggregator.aggregate_scores(
                sample_evaluations, AggregationStrategy.MOST_RESPECTED_PERSON
            )

    def test_borda_count_error_pytest(self, aggregator, sample_evaluations):
        """Test Borda count error handling using pytest."""
        with pytest.raises(ValueError, match="Rankings required for Borda Count"):
            aggregator.aggregate_scores(
                sample_evaluations, AggregationStrategy.BORDA_COUNT
            )

    @pytest.mark.parametrize(
        "strategy,expected_top",
        [
            (AggregationStrategy.LEAST_MISERY, "item_A"),
            (AggregationStrategy.MOST_PLEASURE, "item_C"),
            (AggregationStrategy.ADDITIVE_UTILITARIAN, "item_A"),
        ],
    )
    def test_top_recommendation_parametrized(
        self, aggregator, sample_evaluations, strategy, expected_top
    ):
        """Test top recommendations with parametrized testing."""
        top_item = aggregator.get_top_recommendation(sample_evaluations, strategy)
        assert top_item == expected_top

    def test_empty_evaluations_pytest(self, aggregator):
        """Test empty evaluations using pytest."""
        result = aggregator.aggregate_scores({}, AggregationStrategy.AVG_PREDICTIONS)
        assert result == {}

    def test_multiplicative_with_zero(self, aggregator):
        """Test multiplicative aggregation with zero values."""
        evaluations_with_zero = {
            "user1": {"item_A": 0.0, "item_B": 3.0},
            "user2": {"item_A": 4.0, "item_B": 2.0},
        }

        result = aggregator.aggregate_scores(
            evaluations_with_zero, AggregationStrategy.MULTIPLICATIVE
        )

        assert result["item_A"] == 0.0  # 0.0 * 4.0 = 0.0
        assert result["item_B"] == 6.0  # 3.0 * 2.0 = 6.0


# Integration tests
class TestScoreAggregatorIntegration(unittest.TestCase):
    """Integration tests for ScoreAggregator."""

    def test_all_strategies_consistency(self):
        """Test that all strategies produce consistent results."""
        evaluations = {
            "user1": {"item_A": 5.0, "item_B": 3.0, "item_C": 4.0},
            "user2": {"item_A": 4.0, "item_B": 5.0, "item_C": 3.0},
            "user3": {"item_A": 3.0, "item_B": 4.0, "item_C": 5.0},
        }

        rankings = {
            "user1": ["item_A", "item_C", "item_B"],
            "user2": ["item_B", "item_A", "item_C"],
            "user3": ["item_C", "item_B", "item_A"],
        }

        aggregator = ScoreAggregator(most_respected_person="user1")

        # Test that all strategies return valid results
        strategies_without_rankings = [
            AggregationStrategy.AVG_PREDICTIONS,
            AggregationStrategy.LEAST_MISERY,
            AggregationStrategy.MOST_PLEASURE,
            AggregationStrategy.MOST_RESPECTED_PERSON,
            AggregationStrategy.AVG_PREFERENCES,
            AggregationStrategy.ADDITIVE_UTILITARIAN,
            AggregationStrategy.MULTIPLICATIVE,
        ]

        for strategy in strategies_without_rankings:
            result = aggregator.aggregate_scores(evaluations, strategy)  # type: ignore
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 3)  # Should have 3 items
            for score in result.values():
                self.assertIsInstance(score, (int, float))

        # Test Borda count separately
        borda_result = aggregator.aggregate_scores(
            evaluations,  # type: ignore
            AggregationStrategy.BORDA_COUNT,
            rankings,  # type: ignore
        )
        self.assertIsInstance(borda_result, dict)
        self.assertEqual(len(borda_result), 3)


if __name__ == "__main__":
    # Run unittest tests
    unittest.main(verbosity=2)
