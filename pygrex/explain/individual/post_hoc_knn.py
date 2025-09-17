from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, Any

from .explainer import Explainer


class KNNPostHocExplainer(Explainer):
    def __init__(self, model, recommendations, data, knn=10):
        super(KNNPostHocExplainer, self).__init__(model, recommendations, data)

        self.knn = knn
        # Initialize as an empty dictionary to prevent subscripting None
        self.knn_items_dict: Dict[int, np.ndarray] = {}

    def get_nn_for_getting(self, item_id: int) -> np.ndarray:
        # Check if the KNN dictionary has been computed
        if not self.knn_items_dict:
            self.compute_knn_items_for_all_items()

        # Return the neighbors for the item, or an empty array if not found
        return self.knn_items_dict.get(item_id, np.array([]))

    def compute_knn_items_for_all_items(self):
        ds = np.zeros((self.num_items, self.num_users))
        # Assuming self.dataset has attributes itemId, userId, and rating
        ds[self.dataset.itemId, self.dataset.userId] = self.dataset.rating

        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.num_items):
            sim_matrix[i, i] = min_val
            knn_to_item_i = (-sim_matrix[i, :]).argsort()[: self.knn]
            self.knn_items_dict[i] = knn_to_item_i

    def explain_recommendation_to_user(
        self, user_id: int, item_id: int
    ) -> Dict[str, Any]:
        user_ratings = self.get_user_items(user_id)
        sim_items = self.get_nn_for_getting(item_id)
        explanations = set(sim_items) & set(user_ratings)

        return {"explanations": explanations}
