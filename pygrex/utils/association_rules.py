from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
from pygrex.data_reader.data_reader import DataReader
from typing import List, Optional, Union


class AssociationRules:
    """
    A class to represent association rules mining for recommendation systems.

    This class implements association rules mining using the FP-Growth algorithm
    to discover frequent itemsets and generate association rules from user-item
    interaction data. It can be used to find patterns in user behavior and
    generate item recommendations based on item associations.
    """

    def __init__(
        self,
        data: DataReader,
        min_support: float = 0.2,
        min_confidence: float = 0.2,
        rating_threshold: float = 4.0,
    ) -> None:
        """Initialize the association rules miner with data and parameters.

        Args:
            data: The DataReader object containing user-item interactions with ratings.
            min_support: Minimum support threshold for frequent itemsets.
                Must be between 0 and 1. Default is 0.2.
            min_confidence: Minimum confidence threshold for association rules.
                Must be between 0 and 1. Default is 0.2.
            rating_threshold: Minimum rating threshold to consider an interaction
                as positive. Default is 4.0.

        Raises:
            ValueError: If support, confidence, or rating_threshold values are invalid.
        """
        self._validate_parameters(min_support, min_confidence, rating_threshold)

        self.data = data
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rating_threshold = rating_threshold
        self._frequent_itemsets: Optional[pd.DataFrame] = None
        self._association_rules: Optional[pd.DataFrame] = None

    def _validate_parameters(
        self, min_support: float, min_confidence: float, rating_threshold: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            min_support: Minimum support threshold to validate.
            min_confidence: Minimum confidence threshold to validate.
            rating_threshold: Rating threshold to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not (0 < min_support <= 1):
            raise ValueError("min_support must be between 0 and 1")
        if not (0 < min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")
        if rating_threshold < 0:
            raise ValueError("rating_threshold must be non-negative")

    def get_df_filtered_by_rating_threshold(self) -> pd.DataFrame:
        df = self.data.dataset.copy()
        # Filter interactions based on rating threshold
        df_filtered = df[df["rating"] >= self.rating_threshold]

        if df_filtered.empty:
            raise ValueError(
                f"No interactions found with rating >= {self.rating_threshold}"
            )
        return df_filtered

    def _prepare_transactions(self) -> List[List[str]]:
        """Prepare transaction data from the dataset.

        Filters the dataset based on rating threshold and groups items
        by user to create transaction lists.

        Returns:
            A list of transactions, where each transaction is a list of item IDs
            that a user has positively interacted with.
        """
        df_filtered = self.get_df_filtered_by_rating_threshold()
        # Group items by user to create transactions
        transactions = df_filtered.groupby("userId")["itemId"].apply(list).tolist()

        # Convert item IDs to strings for consistency
        transactions = [
            [str(item) for item in transaction] for transaction in transactions
        ]

        return transactions

    def _mine_frequent_itemsets(
        self, transactions: List[List[Union[str, int]]]
    ) -> pd.DataFrame:
        """Mine frequent itemsets using FP-Growth algorithm.

        Args:
            transactions: List of transactions to mine frequent itemsets from.

        Returns:
            DataFrame containing frequent itemsets with their support values.

        Raises:
            ValueError: If no frequent itemsets are found.
        """
        # Encode transactions into binary matrix
        transaction_encoder = TransactionEncoder()
        transaction_matrix = transaction_encoder.fit_transform(transactions)

        df_encoded = pd.DataFrame(
            transaction_matrix,  # type: ignore
            columns=transaction_encoder.columns_,
        )

        # Apply FP-Growth to find frequent itemsets
        frequent_itemsets = fpgrowth(
            df_encoded, min_support=self.min_support, use_colnames=True
        )

        if frequent_itemsets.empty:
            raise ValueError(
                f"No frequent itemsets found with min_support={self.min_support}"
            )

        return frequent_itemsets

    def _generate_association_rules(
        self, frequent_itemsets: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate association rules from frequent itemsets.

        Args:
            frequent_itemsets: DataFrame containing frequent itemsets.

        Returns:
            DataFrame containing association rules with their metrics.

        Raises:
            ValueError: If no association rules are found.
        """
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=self.min_confidence
        )

        if rules.empty:
            raise ValueError(
                f"No association rules found with min_confidence={self.min_confidence}"
            )

        return rules

    def compute(self) -> pd.DataFrame:
        """Compute association rules from the dataset.

        This method performs the complete association rules mining process:
        1. Prepares transactions from the dataset
        2. Mines frequent itemsets using FP-Growth
        3. Generates association rules from frequent itemsets

        Returns:
            DataFrame containing association rules with metrics including
            antecedents, consequents, support, confidence, lift, etc.

        Raises:
            ValueError: If the dataset is empty, no transactions meet the
                criteria, or no rules can be generated with the given parameters.
        """
        if self.data.dataset.empty:
            raise ValueError("Dataset is empty")

        # Prepare transactions
        transactions = self._prepare_transactions()

        if not transactions:
            raise ValueError("No transactions found after filtering")

        # Mine frequent itemsets
        self._frequent_itemsets = self._mine_frequent_itemsets(transactions)  # type: ignore

        # Generate association rules
        self._association_rules = self._generate_association_rules(
            self._frequent_itemsets
        )

        return self._association_rules

    def get_frequent_itemsets(self) -> Optional[pd.DataFrame]:
        """Get the computed frequent itemsets.

        Returns:
            DataFrame containing frequent itemsets if compute() has been called,
            None otherwise.
        """
        return self._frequent_itemsets

    def get_recommendations_for_items(
        self, items: List[Union[str, int]], top_k: int = 10
    ) -> pd.DataFrame:
        """Get item recommendations based on association rules.

        Args:
            items: List of item IDs to get recommendations for.
            top_k: Maximum number of recommendations to return. Default is 10.

        Returns:
            DataFrame containing recommended items sorted by confidence.

        Raises:
            RuntimeError: If compute() hasn't been called yet.
            ValueError: If items list is empty.
        """
        if self._association_rules is None:
            raise RuntimeError("Must call compute() before getting recommendations")

        if not items:
            raise ValueError("Items list cannot be empty")

        items_set = set(str(item) for item in items)

        # Filter rules where antecedents match the given items
        matching_rules = self._association_rules[
            self._association_rules["antecedents"].apply(
                lambda x: items_set.issubset(set(str(item) for item in x))
            )
        ]

        if matching_rules.empty:
            return pd.DataFrame()

        # Sort by confidence and return top_k recommendations
        recommendations = matching_rules.nlargest(top_k, "confidence")

        return recommendations[
            ["antecedents", "consequents", "confidence", "lift", "support"]
        ]

    def __str__(self) -> str:
        """Return string representation of the AssociationRules object."""
        return (
            f"AssociationRules(min_support={self.min_support}, "
            f"min_confidence={self.min_confidence}, "
            f"rating_threshold={self.rating_threshold})"
        )

    def __repr__(self) -> str:
        """Return detailed string representation of the AssociationRules object."""
        return self.__str__()
