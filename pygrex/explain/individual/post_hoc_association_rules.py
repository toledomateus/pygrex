from typing import Any, Dict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

from .explainer import Explainer


class ARPostHocExplainer(Explainer):
    def __init__(
        self,
        model,
        recommendations,
        data,
        min_support=0.1,
        max_len=2,
        metric="lift",
        min_threshold=0.1,
        min_confidence=0.1,
        min_lift=0.1,
    ):
        super(ARPostHocExplainer, self).__init__(model, recommendations, data)
        self.AR = None
        self.min_support = min_support
        self.max_len = max_len
        self.metric = metric
        self.min_threshold = min_threshold
        self.min_confidence = min_confidence
        self.min_lift = min_lift

        self.rules: pd.DataFrame | None = None

    def get_rules_for_getting(self, item_id: int) -> pd.DataFrame:
        if self.rules is None:
            self.compute_association_rules()

        if self.rules is not None:
            return self.rules[self.rules.consequents == item_id]

        return pd.DataFrame()

    def compute_association_rules(self):
        item_sets = [
            [item for item in self.dataset[self.dataset.userId == user].itemId]
            for user in self.dataset.userId.unique()
        ]

        te = TransactionEncoder()
        te_ary = te.fit(item_sets).transform(item_sets)

        # The te_ary object is a NumPy array, which is a valid input for a DataFrame.
        # Pylance may raise a false positive here due to incomplete type stubs for mlxtend.
        df = pd.DataFrame(te_ary.astype(bool), columns=te.columns_)  # type: ignore

        frequent_itemsets = apriori(
            df, min_support=self.min_support, use_colnames=True, max_len=self.max_len
        )

        rules = association_rules(
            frequent_itemsets, metric="lift", min_threshold=self.min_threshold
        )
        rules = rules[
            (rules["confidence"] > self.min_confidence)
            & (rules["lift"] > self.min_lift)
        ]

        rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0])
        rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0])

        self.rules = rules[["consequents", "antecedents", "confidence"]]

    def explain_recommendation_to_user(
        self, user_id: int, item_id: int
    ) -> Dict[str, Any]:
        user_ratings = self.get_user_items(user_id)
        rules = self.get_rules_for_getting(item_id)
        explanations = rules[rules.antecedents.isin(user_ratings)]

        return {"antecedents": set(explanations.antecedents)}
