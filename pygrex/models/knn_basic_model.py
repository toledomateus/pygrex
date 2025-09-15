import numpy as np
import scipy.sparse as sp
from .recommender_model import RecommenderModel
from ..data_reader import DataReader


class KNNBasic(RecommenderModel):
    """
    An improved K-Nearest Neighbors collaborative filtering model.

    This version uses Pearson correlation similarity and improved neighbor selection
    for better performance on sparse datasets like MovieLens.

    Args:
        k (int): Number of neighbors to consider. Default 50.
        min_k (int): Minimum number of neighbors required for prediction. Default 3.
        sim_options (dict): Similarity options. Default pearson, user-based.
    """

    def __init__(self, k: int = 50, min_k: int = 3, sim_options: dict = None):
        super().__init__()
        self.k = k
        self.min_k = min_k
        self.sim_options = sim_options if sim_options is not None else {}

        # Validate similarity options
        if self.sim_options.get("user_based", True) is False:
            raise NotImplementedError("Only the user-based approach is implemented.")

        sim_name = self.sim_options.get("name", "pearson").lower()
        if sim_name not in ["cosine", "pearson"]:
            raise NotImplementedError(
                "Only cosine and pearson similarity are implemented."
            )

        # Model attributes
        self.trainset = None
        self.global_mean = 0
        self.user_biases = None
        self.item_biases = None
        self.num_users = None
        self.num_items = None

        # For memory-efficient similarity computation
        self.user_means = None

    def fit(self, dataset: DataReader) -> None:
        """
        Trains the KNN model with improved memory efficiency.
        """
        print("Fitting the improved KNNBasic model...")
        df = dataset.dataset
        self.num_users = dataset.num_user
        self.num_items = dataset.num_item

        print(
            f"Building ratings matrix for {self.num_users} users and {self.num_items} items..."
        )

        # 1. Build the sparse user-item ratings matrix
        ratings = df["rating"].values
        rows = df["userId"].values
        cols = df["itemId"].values
        self.trainset = sp.csr_matrix(
            (ratings, (rows, cols)), shape=(self.num_users, self.num_items)
        )

        # 2. Calculate global mean and biases
        print("Computing biases...")
        self.global_mean = self.trainset.data.mean()

        # User biases: bu = avg(ratings_u) - global_mean
        user_sums = np.array(self.trainset.sum(axis=1)).flatten()
        user_counts = np.diff(self.trainset.indptr)

        with np.errstate(divide="ignore", invalid="ignore"):
            user_avg_ratings = np.where(
                user_counts > 0, user_sums / user_counts, self.global_mean
            )
        self.user_biases = np.where(
            user_counts > 0, user_avg_ratings - self.global_mean, 0
        )

        # Item biases: bi = avg(ratings_i) - global_mean
        item_sums = np.array(self.trainset.sum(axis=0)).flatten()
        item_counts = np.diff(self.trainset.tocsc().indptr)

        with np.errstate(divide="ignore", invalid="ignore"):
            item_avg_ratings = np.where(
                item_counts > 0, item_sums / item_counts, self.global_mean
            )
        self.item_biases = np.where(
            item_counts > 0, item_avg_ratings - self.global_mean, 0
        )

        # Store user means for similarity computation
        self.user_means = user_avg_ratings

        print("Model fitting complete.")

    def _compute_user_similarity(self, user1_id: int, user2_id: int) -> float:
        """
        Compute Pearson correlation similarity between two users.
        This works better than cosine similarity for collaborative filtering.
        """
        # Get rating vectors for both users
        user1_ratings = self.trainset[user1_id].toarray().flatten()
        user2_ratings = self.trainset[user2_id].toarray().flatten()

        # Find commonly rated items
        mask = (user1_ratings > 0) & (user2_ratings > 0)
        n_common = np.sum(mask)

        # Need at least 2 common ratings for correlation
        if n_common < 2:
            return 0.0

        # Extract ratings for commonly rated items
        u1_common = user1_ratings[mask]
        u2_common = user2_ratings[mask]

        # Mean-center the ratings
        u1_mean = np.mean(u1_common)
        u2_mean = np.mean(u2_common)

        u1_centered = u1_common - u1_mean
        u2_centered = u2_common - u2_mean

        # Compute Pearson correlation
        numerator = np.sum(u1_centered * u2_centered)
        denom1 = np.sqrt(np.sum(u1_centered**2))
        denom2 = np.sqrt(np.sum(u2_centered**2))

        if denom1 == 0 or denom2 == 0:
            return 0.0

        correlation = numerator / (denom1 * denom2)

        # Apply significance weighting based on number of common items
        # More common items = more reliable similarity
        significance_weight = min(n_common / 50.0, 1.0)  # Cap at 50 common items

        return correlation * significance_weight

    def _get_neighbors_for_item(self, user_id: int, item_id: int):
        """
        Get the top-k most similar users who have rated the given item.
        """
        # Find users who rated this item
        item_col = self.trainset[:, item_id]  # type: ignore
        neighbor_candidates, _ = item_col.nonzero()

        # Remove the target user if they're in the candidates
        neighbor_candidates = neighbor_candidates[neighbor_candidates != user_id]

        if len(neighbor_candidates) == 0:
            return [], [], []

        # Compute similarities
        similarities = []
        for neighbor_id in neighbor_candidates:
            sim = self._compute_user_similarity(user_id, neighbor_id)
            similarities.append((sim, neighbor_id))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[: min(self.k, len(similarities))]

        if len(top_k) < self.min_k:
            return [], [], []

        # Extract data
        neighbor_sims = np.array([sim for sim, _ in top_k])
        neighbor_ids = np.array([nid for _, nid in top_k])
        neighbor_ratings = np.array(
            [self.trainset[nid, item_id] for nid in neighbor_ids]
        )

        return neighbor_sims, neighbor_ids, neighbor_ratings

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using KNN.
        """
        if self.trainset is None:
            raise RuntimeError("Model must be trained first using fit() method.")

        # Handle out-of-bounds users/items
        if user_id >= self.num_users or item_id >= self.num_items:
            return self.global_mean

        # 1. Calculate baseline estimate
        baseline = (
            self.global_mean + self.user_biases[user_id] + self.item_biases[item_id]
        )

        # 2. Get neighbors who rated this item
        neighbor_sims, neighbor_ids, neighbor_ratings = self._get_neighbors_for_item(
            user_id, item_id
        )

        if len(neighbor_ids) == 0:
            return baseline

        # 3. Calculate weighted prediction
        neighbor_biases = self.user_biases[neighbor_ids]
        neighbor_baselines = (
            self.global_mean + neighbor_biases + self.item_biases[item_id]
        )

        deviations = neighbor_ratings - neighbor_baselines

        # Only use neighbors with positive similarity
        positive_mask = neighbor_sims > 0
        if not np.any(positive_mask):
            return baseline

        neighbor_sims = neighbor_sims[positive_mask]
        deviations = deviations[positive_mask]

        numerator = np.sum(neighbor_sims * deviations)
        denominator = np.sum(np.abs(neighbor_sims))

        if denominator == 0:
            return baseline

        prediction = baseline + (numerator / denominator)

        # Clip to valid rating range
        return np.clip(prediction, 1.0, 5.0)
