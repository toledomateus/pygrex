"""Rule-based group recommendation explainer module."""

from typing import Dict, List, Optional, Set, Union
import logging

from pygrex.data_reader.data_reader import DataReader
from pygrex.utils.association_rules import AssociationRules

# Type aliases for better readability
ItemId = Union[str, int]
MemberId = Union[str, int]
UserHistory = Dict[MemberId, Set[ItemId]]

logger = logging.getLogger(__name__)


class RuleBasedGroupRecExplainer:
    """
    A class to explain group recommendations using rule-based methods.

    This class provides methods to generate explanations for group recommendations
    based on association rules and user interaction history.
    """

    def __init__(
        self,
        rules: AssociationRules,
        data: DataReader,
        pool_recommendations: Optional[Union[List[ItemId], ItemId]] = None,
        members: Optional[List[MemberId]] = None,
        user_history: Optional[UserHistory] = None,
        min_members_threshold: int = 1,
    ) -> None:
        """
        Initialize the RuleBasedGroupRecExplainer.

        Args:
            rules: An instance of AssociationRules containing the rules for explanations.
            pool_recommendations: A list of item IDs to explain, or a single item ID.
            members: A list of member IDs in the group.
            user_history: A dictionary mapping member IDs to sets of item IDs
                         they have interacted with.
            min_members_threshold: Minimum number of members that must satisfy
                                 the rule for it to be considered valid.

        Raises:
            ValueError: If min_members_threshold is less than 1.
        """
        if min_members_threshold < 1:
            raise ValueError("min_members_threshold must be at least 1")

        self.rules = rules
        self.members = members or []
        self.min_members_threshold = min_members_threshold
        self.user_history = user_history or {}
        self.data = data

        # Normalize pool_recommendations to always be a list
        self.pool_recommendations = self._normalize_recommendations(
            pool_recommendations
        )

    def _normalize_recommendations(
        self, recommendations: Optional[Union[List[ItemId], ItemId]]
    ) -> List[ItemId]:
        """
        Normalize recommendations input to a list format.

        Args:
            recommendations: Single item ID, list of item IDs, or None.

        Returns:
            List of item IDs.
        """
        if recommendations is None:
            return []

        if isinstance(recommendations, (str, int)):
            return [recommendations]

        return recommendations

    def _is_rule_satisfied_by_member(
        self, member: MemberId, antecedent: Set[ItemId]
    ) -> bool:
        """
        Check if a member satisfies the rule's antecedent.

        Args:
            member: The member ID to check.
            antecedent: The set of items that form the rule's antecedent.

        Returns:
            True if the member's history contains all items in the antecedent.
        """

        member_history = self.user_history.get(member, set())
        member_history_str = {str(item) for item in member_history}

        x = member_history_str.issuperset(antecedent)
        return x

    def _count_satisfied_members(self, antecedent: Set[ItemId]) -> int:
        """
        Count how many members satisfy the given antecedent.

        Args:
            antecedent: The set of items that form the rule's antecedent.

        Returns:
            Number of members whose history satisfies the antecedent.
        """
        return sum(
            1
            for member in self.members
            if self._is_rule_satisfied_by_member(member, antecedent)
        )

    def _find_applicable_rules(self, item_id: ItemId):
        """
        Find rules that have the given item in their consequents.

        Args:
            item_id: The item ID to find rules for.

        Returns:
            DataFrame containing applicable rules.
        """
        item_id = self.data.get_new_item_id(item_id)  # type: ignore

        applicable_rules = self.rules[  # type: ignore
            self.rules["consequents"].apply(lambda x: str(item_id) in x)  # type: ignore
        ]

        return applicable_rules

    def find_explanation(self) -> float:
        """
        Generate explanations for the group recommendations based on the rules.

        Returns:
            The fidelity of the explanations, which is the ratio of explained
            recommendations to total recommendations in the pool.
        """
        if not self.pool_recommendations:
            logger.warning("No recommendations to explain")
            return 0.0

        explained_count = 0
        total_recommendations = len(self.pool_recommendations)

        for item_id in self.pool_recommendations:
            if self._can_explain_item(item_id):
                explained_count += 1

        fidelity = explained_count / total_recommendations
        logger.info(
            f"Explained {explained_count}/{total_recommendations} recommendations "
            f"(fidelity: {fidelity:.3f})"
        )

        return fidelity

    def _can_explain_item(self, item_id: ItemId) -> bool:
        """
        Check if an item can be explained by any rule.

        Args:
            item_id: The item ID to check.

        Returns:
            True if at least one rule can explain the item.
        """
        applicable_rules = self._find_applicable_rules(item_id)

        for _, rule in applicable_rules.iterrows():
            antecedent = rule["antecedents"]
            satisfied_count = self._count_satisfied_members(antecedent)

            if satisfied_count >= self.min_members_threshold:
                logger.debug(f"Rule fired for item {item_id}")
                return True

        return False

    def get_explanation_details(self) -> Dict[ItemId, List[Dict]]:
        """
        Get detailed explanations for each recommendation.

        Returns:
            Dictionary mapping item IDs to lists of applicable rule details.
        """
        explanations = {}

        for item_id in self.pool_recommendations:
            item_explanations = []
            applicable_rules = self._find_applicable_rules(item_id)

            for _, rule in applicable_rules.iterrows():
                antecedent = rule["antecedents"]
                satisfied_count = self._count_satisfied_members(antecedent)

                if satisfied_count >= self.min_members_threshold:
                    item_explanations.append(
                        {
                            "antecedent": antecedent,
                            "consequent": rule["consequents"],
                            "satisfied_members": satisfied_count,
                            "confidence": rule.get("confidence", "N/A"),
                            "support": rule.get("support", "N/A"),
                        }
                    )

            explanations[item_id] = item_explanations

        return explanations

    def compute_group_fidelity_advanced(self) -> float:
        """
        Compute group fidelity using advanced conditions.

        This method implements a more sophisticated fidelity calculation where:
        - Condition 1: Each member of the group must have seen at least one item from the antecedent
        - Condition 2: Each item in the antecedent must have been seen by at least one member

        Returns:
            The fidelity score as a float between 0 and 1.
        """
        if not self.pool_recommendations:
            logger.warning("No recommendations to explain")
            return 0.0

        if not self.members:
            logger.warning("No group members defined")
            return 0.0

        explained_count = 0
        total_recommendations = len(self.pool_recommendations)

        # Convert member IDs to set for faster lookup
        members_set = set(self.members)

        # Get all items seen by any group member
        all_seen_items = set()
        for member in members_set:
            member_history = self.user_history.get(member, set())
            # Convert to strings for consistency with rules
            member_history_str = {str(item) for item in member_history}
            all_seen_items.update(member_history_str)

        for item_id in self.pool_recommendations:
            if self._can_explain_item_advanced(item_id, members_set, all_seen_items):
                explained_count += 1

        fidelity = explained_count / total_recommendations
        logger.info(
            f"Advanced explanation: {explained_count}/{total_recommendations} recommendations "
            f"(fidelity: {fidelity:.3f})"
        )

        return fidelity

    def _can_explain_item_advanced(
        self, item_id: ItemId, members_set: Set[MemberId], all_seen_items: Set[str]
    ) -> bool:
        """
        Check if an item can be explained using advanced conditions.

        Args:
            item_id: The item ID to check.
            members_set: Set of group member IDs.
            all_seen_items: Set of all items seen by any group member.

        Returns:
            True if the item can be explained by at least one rule satisfying both conditions.
        """
        applicable_rules = self._find_applicable_rules(item_id)

        for _, rule in applicable_rules.iterrows():
            antecedent = rule["antecedents"]

            # Condition 1: Each member must have seen at least one item from the antecedent
            cond1 = all(
                self._member_has_antecedent_item(member, antecedent)
                for member in members_set
            )

            # Condition 2: Each item in the antecedent must have been seen by at least one member
            cond2 = antecedent.issubset(all_seen_items)

            if cond1 and cond2:
                logger.debug(f"Advanced rule fired for item {item_id}")
                return True

        return False

    def _member_has_antecedent_item(
        self, member: MemberId, antecedent: Set[ItemId]
    ) -> bool:
        """
        Check if a member has seen at least one item from the antecedent.

        Args:
            member: The member ID to check.
            antecedent: The set of items in the rule's antecedent.

        Returns:
            True if the member has seen at least one item from the antecedent.
        """
        member_history = self.user_history.get(member, set())
        member_history_str = {str(item) for item in member_history}

        # Check if there's any intersection between member history and antecedent
        return len(antecedent.intersection(member_history_str)) > 0
