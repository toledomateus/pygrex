import numpy as np
from typing import Dict, List, Union, Optional, TypeAlias
from enum import Enum

# Type aliases for better readability
UserID: TypeAlias = Union[str, int]
ItemID: TypeAlias = Union[str, int]
EvaluationScore: TypeAlias = float
AggregatedScore: TypeAlias = float

# Main data structure types
UserEvaluations: TypeAlias = Dict[UserID, Dict[ItemID, EvaluationScore]]
UserRankings: TypeAlias = Dict[UserID, List[ItemID]]
AggregatedScores: TypeAlias = Dict[ItemID, AggregatedScore]


class AggregationStrategy(Enum):
    """Enumeration of available aggregation strategies."""

    # Individual Predictions
    AVG_PREDICTIONS = "avg_predictions"
    LEAST_MISERY = "least_misery"
    MOST_PLEASURE = "most_pleasure"
    MOST_RESPECTED_PERSON = "most_respected_person"

    # Individual Preferences
    ADDITIVE_UTILITARIAN = "additive_utilitarian"
    MULTIPLICATIVE = "multiplicative"
    BORDA_COUNT = "borda_count"


class ScoreAggregator:
    """
    A class for aggregating individual predictions or preferences into collective scores.

    Supports two main approaches:
    1. Individual Predictions: AVG, LM, MP, MRP
    2. Individual Preferences: AVG, ADD, MUL, BRC

    Felfernig, A., Boratto, L., Stettinger, M., Tkali, M.: Group Recommender Systems:
    An Introduction. Springer Publishing Company, Incorporated, 1st edn. (2018)

    """

    def __init__(self, most_respected_person: Optional[UserID] = None):
        """
        Initialize the ScoreAggregator.

        Args:
            most_respected_person: User ID of the most respected person (required for MRP strategy)
        """
        self.most_respected_person = most_respected_person

    def aggregate_scores(
        self,
        evaluations: UserEvaluations,
        strategy: AggregationStrategy,
        rankings: Optional[UserRankings] = None,
    ) -> AggregatedScores:
        """
        Aggregate individual evaluations into collective scores.

        Args:
            evaluations: Dictionary mapping user_id -> {item_id: evaluation_score}
            strategy: Aggregation strategy to use
            rankings: Dictionary mapping user_id -> [ordered_list_of_items] (required for Borda Count)

        Returns:
            Dictionary mapping item_id -> aggregated_score
        """
        if not evaluations:
            return {}

        # Get all items across all users
        all_items: set[ItemID] = set()
        for user_evals in evaluations.values():
            all_items.update(user_evals.keys())

        result: AggregatedScores = {}

        for item in all_items:
            if strategy == AggregationStrategy.AVG_PREDICTIONS:
                result[item] = self._avg_predictions(evaluations, item)
            elif strategy == AggregationStrategy.LEAST_MISERY:
                result[item] = self._least_misery(evaluations, item)
            elif strategy == AggregationStrategy.MOST_PLEASURE:
                result[item] = self._most_pleasure(evaluations, item)
            elif strategy == AggregationStrategy.MOST_RESPECTED_PERSON:
                result[item] = self._most_respected_person(evaluations, item)
            elif strategy == AggregationStrategy.ADDITIVE_UTILITARIAN:
                result[item] = self._additive_utilitarian(evaluations, item)
            elif strategy == AggregationStrategy.MULTIPLICATIVE:
                result[item] = self._multiplicative(evaluations, item)
            elif strategy == AggregationStrategy.BORDA_COUNT:
                if rankings is None:
                    raise ValueError("Rankings required for Borda Count strategy")
                result[item] = self._borda_count(rankings, item)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")

        return result

    def get_top_recommendation(
        self,
        evaluations: UserEvaluations,
        strategy: AggregationStrategy,
        rankings: Optional[UserRankings] = None,
    ) -> ItemID:
        """
        Get the top recommended item based on aggregated scores.

        Args:
            evaluations: Dictionary mapping user_id -> {item_id: evaluation_score}
            strategy: Aggregation strategy to use
            rankings: Dictionary mapping user_id -> [ordered_list_of_items] (required for Borda Count)

        Returns:
            Item ID with highest aggregated score
        """
        aggregated_scores = self.aggregate_scores(evaluations, strategy, rankings)
        return max(aggregated_scores.items(), key=lambda x: x[1])[0]

    def _avg_predictions(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Average of item-specific evaluations."""
        item_evals = [
            user_evals.get(item, 0)
            for user_evals in evaluations.values()
            if item in user_evals
        ]
        return np.mean(item_evals) if item_evals else 0.0  # type: ignore

    def _least_misery(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Minimum item-specific evaluation."""
        item_evals = [
            user_evals.get(item, 0)
            for user_evals in evaluations.values()
            if item in user_evals
        ]
        return min(item_evals) if item_evals else 0.0

    def _most_pleasure(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Maximum item-specific evaluation."""
        item_evals = [
            user_evals.get(item, 0)
            for user_evals in evaluations.values()
            if item in user_evals
        ]
        return max(item_evals) if item_evals else 0.0

    def _most_respected_person(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Item-evaluations of most respected user."""
        if self.most_respected_person is None:
            raise ValueError("Most respected person not specified")
        if self.most_respected_person not in evaluations:
            raise ValueError(
                f"Most respected person '{self.most_respected_person}' not found in evaluations"
            )
        return evaluations[self.most_respected_person].get(item, 0.0)

    def _avg_preferences(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Average of item-specific evaluations (same as avg_predictions)."""
        return self._avg_predictions(evaluations, item)

    def _additive_utilitarian(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Sum of item-specific evaluations."""
        item_evals = [
            user_evals.get(item, 0)
            for user_evals in evaluations.values()
            if item in user_evals
        ]
        return sum(item_evals)

    def _multiplicative(
        self, evaluations: UserEvaluations, item: ItemID
    ) -> AggregatedScore:
        """Multiplication of item-specific evaluations."""
        item_evals = [
            user_evals.get(item, 0)
            for user_evals in evaluations.values()
            if item in user_evals
        ]
        if not item_evals:
            return 0.0
        result = 1.0
        for eval_score in item_evals:
            result *= eval_score
        return result

    def _borda_count(self, rankings: UserRankings, item: ItemID) -> AggregatedScore:
        """Sum of item-specific scores derived from item ranking."""
        total_score = 0.0
        for user_ranking in rankings.values():
            if item in user_ranking:
                # Score is based on position in ranking (higher position = higher score)
                position = user_ranking.index(item)
                score = len(user_ranking) - position - 1  # Reverse position for score
                total_score += score
        return total_score
