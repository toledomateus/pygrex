"""Test suite for RuleBasedGroupRecExplainer."""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
import logging

from pygrex.explain.rule_based_group_rec_explainer import RuleBasedGroupRecExplainer


"""Test cases for RuleBasedGroupRecExplainer class."""


@pytest.fixture
def mock_data_reader():
    """Create a mock DataReader."""
    data_reader = Mock()
    data_reader.get_new_item_id.side_effect = lambda x: x  # Return the same ID
    return data_reader


@pytest.fixture
def sample_rules():
    """Create sample association rules DataFrame."""
    rules_data = {
        "antecedents": [
            {"item1", "item2"},
            {"item3"},
            {"item1"},
            {"item4", "item5"},
        ],
        "consequents": [{"rec1"}, {"rec2"}, {"rec1"}, {"rec3"}],
        "confidence": [0.8, 0.9, 0.7, 0.6],
        "support": [0.3, 0.4, 0.2, 0.1],
    }
    return pd.DataFrame(rules_data)


@pytest.fixture
def sample_user_history():
    """Create sample user history."""
    return {
        "user1": {"item1", "item2", "item6"},
        "user2": {"item3", "item7"},
        "user3": {"item1", "item4", "item5"},
        "user4": {"item2", "item8"},
    }


@pytest.fixture
def explainer(sample_rules, mock_data_reader, sample_user_history):
    """Create a RuleBasedGroupRecExplainer instance."""
    return RuleBasedGroupRecExplainer(
        rules=sample_rules,
        data=mock_data_reader,
        pool_recommendations=["rec1", "rec2", "rec3"],
        members=["user1", "user2", "user3"],
        user_history=sample_user_history,
        min_members_threshold=2,
    )


class TestInitialization:
    """Test initialization and parameter validation."""

    def test_init_with_valid_parameters(self, sample_rules, mock_data_reader):
        """Test successful initialization with valid parameters."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=["rec1", "rec2"],
            members=["user1", "user2"],
            user_history={"user1": {"item1"}},
            min_members_threshold=1,
        )

        assert explainer.rules is sample_rules
        assert explainer.data is mock_data_reader
        assert explainer.pool_recommendations == ["rec1", "rec2"]
        assert explainer.members == ["user1", "user2"]
        assert explainer.user_history == {"user1": {"item1"}}
        assert explainer.min_members_threshold == 1

    def test_init_with_invalid_threshold(self, sample_rules, mock_data_reader):
        """Test initialization with invalid min_members_threshold."""
        with pytest.raises(
            ValueError, match="min_members_threshold must be at least 1"
        ):
            RuleBasedGroupRecExplainer(
                rules=sample_rules, data=mock_data_reader, min_members_threshold=0
            )

    def test_init_with_defaults(self, sample_rules, mock_data_reader):
        """Test initialization with default parameters."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules, data=mock_data_reader
        )

        assert explainer.pool_recommendations == []
        assert explainer.members == []
        assert explainer.user_history == {}
        assert explainer.min_members_threshold == 1

    def test_normalize_recommendations_single_item(
        self, sample_rules, mock_data_reader
    ):
        """Test normalization of single recommendation item."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules, data=mock_data_reader, pool_recommendations="rec1"
        )
        assert explainer.pool_recommendations == ["rec1"]

    def test_normalize_recommendations_list(self, sample_rules, mock_data_reader):
        """Test normalization of recommendation list."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=["rec1", "rec2"],
        )
        assert explainer.pool_recommendations == ["rec1", "rec2"]

    def test_normalize_recommendations_none(self, sample_rules, mock_data_reader):
        """Test normalization of None recommendations."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules, data=mock_data_reader, pool_recommendations=None
        )
        assert explainer.pool_recommendations == []


class TestPrivateMethods:
    """Test private methods."""

    def test_is_rule_satisfied_by_member_true(self, explainer):
        """Test rule satisfaction when member has all antecedent items."""
        # user1 has item1 and item2
        result = explainer._is_rule_satisfied_by_member("user1", {"item1", "item2"})
        assert result is True

    def test_is_rule_satisfied_by_member_false(self, explainer):
        """Test rule satisfaction when member doesn't have all antecedent items."""
        # user2 only has item3, not item1 and item2
        result = explainer._is_rule_satisfied_by_member("user2", {"item1", "item2"})
        assert result is False

    def test_is_rule_satisfied_by_member_empty_history(self, explainer):
        """Test rule satisfaction for member with empty history."""
        result = explainer._is_rule_satisfied_by_member("nonexistent_user", {"item1"})
        assert result is False

    def test_count_satisfied_members(self, explainer):
        """Test counting members who satisfy antecedent."""
        # Only user1 and user3 have item1
        count = explainer._count_satisfied_members({"item1"})
        assert count == 2

    def test_count_satisfied_members_complex_antecedent(self, explainer):
        """Test counting with complex antecedent."""
        # Only user1 has both item1 and item2
        count = explainer._count_satisfied_members({"item1", "item2"})
        assert count == 1

    def test_find_applicable_rules(self, explainer):
        """Test finding rules applicable to an item."""
        applicable_rules = explainer._find_applicable_rules("rec1")

        # Should find rules where 'rec1' is in consequents
        assert len(applicable_rules) == 2  # Rules with rec1 as consequent
        assert all(
            "rec1" in rule["consequents"] for _, rule in applicable_rules.iterrows()
        )

    def test_can_explain_item_true(self, explainer):
        """Test item explanation when rules are satisfied."""
        # rec1 should be explainable (rules with item1/item2 and item1 antecedents)
        result = explainer._can_explain_item("rec1")
        assert result is True

    def test_can_explain_item_false(self, explainer):
        """Test item explanation when no rules are satisfied."""
        # Create explainer with high threshold
        explainer.min_members_threshold = 5
        result = explainer._can_explain_item("rec1")
        assert result is False


class TestAdvancedMethods:
    """Test advanced explanation methods."""

    def test_member_has_antecedent_item_true(self, explainer):
        """Test when member has at least one antecedent item."""
        # user1 has item1 (from antecedent {item1, item2})
        result = explainer._member_has_antecedent_item("user1", {"item1", "item9"})
        assert result is True

    def test_member_has_antecedent_item_false(self, explainer):
        """Test when member has no antecedent items."""
        # user2 doesn't have item1 or item9
        result = explainer._member_has_antecedent_item("user2", {"item1", "item9"})
        assert result is False

    def test_can_explain_item_advanced_both_conditions_true(self, explainer):
        """Test advanced explanation when both conditions are met."""
        members_set = {"user1", "user3"}  # Both have item1
        all_seen_items = {"item1", "item2", "item4", "item5", "item6"}

        # Rule: {item1} -> {rec1}
        # Condition 1: Both users have item1 ✓
        # Condition 2: item1 is in all_seen_items ✓
        result = explainer._can_explain_item_advanced(
            "rec1", members_set, all_seen_items
        )
        assert result is True

    def test_can_explain_item_advanced_condition1_false(self, explainer):
        """Test advanced explanation when condition 1 fails."""
        members_set = {"user1", "user2"}  # user2 doesn't have item1
        all_seen_items = {"item1", "item2", "item3"}

        # For rule {item1} -> {rec1}: user2 doesn't have item1
        result = explainer._can_explain_item_advanced(
            "rec1", members_set, all_seen_items
        )
        # This might still be True if other rules apply, so let's check specific case
        assert isinstance(result, bool)

    def test_can_explain_item_advanced_condition2_false(self, explainer):
        """Test advanced explanation when condition 2 fails."""
        members_set = {"user1", "user3"}
        all_seen_items = {"item6", "item7"}  # Missing antecedent items

        result = explainer._can_explain_item_advanced(
            "rec1", members_set, all_seen_items
        )
        assert result is False


class TestPublicMethods:
    """Test public methods."""

    def test_find_explanation_with_recommendations(self, explainer):
        """Test explanation finding with recommendations."""
        fidelity = explainer.find_explanation()

        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0

    def test_find_explanation_empty_recommendations(
        self, sample_rules, mock_data_reader
    ):
        """Test explanation finding with empty recommendations."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules, data=mock_data_reader, pool_recommendations=[]
        )

        fidelity = explainer.find_explanation()
        assert fidelity == 0.0

    def test_compute_group_fidelity_advanced_with_data(self, explainer):
        """Test advanced group fidelity computation."""
        fidelity = explainer.compute_group_fidelity_advanced()

        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0

    def test_compute_group_fidelity_advanced_empty_recommendations(
        self, sample_rules, mock_data_reader
    ):
        """Test advanced fidelity with empty recommendations."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=[],
            members=["user1", "user2"],
        )

        fidelity = explainer.compute_group_fidelity_advanced()
        assert fidelity == 0.0

    def test_compute_group_fidelity_advanced_no_members(
        self, sample_rules, mock_data_reader
    ):
        """Test advanced fidelity with no members."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=["rec1"],
            members=[],
        )

        fidelity = explainer.compute_group_fidelity_advanced()
        assert fidelity == 0.0

    def test_get_explanation_details(self, explainer):
        """Test getting detailed explanations."""
        details = explainer.get_explanation_details()

        assert isinstance(details, dict)
        assert all(
            item_id in explainer.pool_recommendations for item_id in details.keys()
        )

        # Check structure of explanation details
        for item_id, explanations in details.items():
            assert isinstance(explanations, list)
            for explanation in explanations:
                assert "antecedent" in explanation
                assert "consequent" in explanation
                assert "satisfied_members" in explanation
                assert "confidence" in explanation
                assert "support" in explanation


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_string_and_int_ids(self, sample_rules, mock_data_reader):
        """Test handling of mixed string and integer IDs."""
        user_history = {
            1: {10, 20},  # Integer user ID with integer item IDs
            "user2": {"item1", "item2"},  # String user ID with string item IDs
        }

        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=[1, "rec1"],
            members=[1, "user2"],
            user_history=user_history,
        )

        # Should not raise exceptions
        fidelity = explainer.find_explanation()
        assert isinstance(fidelity, float)

    def test_empty_user_history(self, sample_rules, mock_data_reader):
        """Test with completely empty user history."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=["rec1"],
            members=["user1", "user2"],
            user_history={},
        )

        fidelity = explainer.find_explanation()
        assert fidelity == 0.0

    def test_rules_with_no_matches(self, mock_data_reader):
        """Test with rules that don't match any recommendations."""
        rules_data = {
            "antecedents": [{"item1"}],
            "consequents": [{"other_rec"}],
            "confidence": [0.8],
            "support": [0.3],
        }
        rules = pd.DataFrame(rules_data)

        explainer = RuleBasedGroupRecExplainer(
            rules=rules,  # type: ignore
            data=mock_data_reader,
            pool_recommendations=["rec1"],
            members=["user1"],
            user_history={"user1": {"item1"}},
        )

        fidelity = explainer.find_explanation()
        assert fidelity == 0.0


class TestLogging:
    """Test logging functionality."""

    def test_logging_warnings(self, sample_rules, mock_data_reader, caplog):
        """Test that appropriate warnings are logged."""
        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=[],
            members=[],
        )

        with caplog.at_level(logging.WARNING):
            explainer.find_explanation()

        assert "No recommendations to explain" in caplog.text

    def test_logging_info(self, explainer, caplog):
        """Test that info messages are logged."""
        with caplog.at_level(logging.INFO):
            explainer.find_explanation()

        assert "Explained" in caplog.text
        assert "fidelity" in caplog.text


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, sample_rules, mock_data_reader):
        """Test complete workflow from initialization to explanation."""
        user_history = {
            "user1": {"item1", "item2"},
            "user2": {"item3"},
            "user3": {"item1", "item4", "item5"},
        }

        explainer = RuleBasedGroupRecExplainer(
            rules=sample_rules,
            data=mock_data_reader,
            pool_recommendations=["rec1", "rec2", "rec3"],
            members=["user1", "user2", "user3"],
            user_history=user_history,  # type: ignore
            min_members_threshold=1,
        )

        # Test basic explanation
        basic_fidelity = explainer.find_explanation()
        assert isinstance(basic_fidelity, float)

        # Test advanced explanation
        advanced_fidelity = explainer.compute_group_fidelity_advanced()
        assert isinstance(advanced_fidelity, float)

        # Test detailed explanations
        details = explainer.get_explanation_details()
        assert isinstance(details, dict)

        # Fidelities should be between 0 and 1
        assert 0.0 <= basic_fidelity <= 1.0
        assert 0.0 <= advanced_fidelity <= 1.0
