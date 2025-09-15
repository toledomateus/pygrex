import numpy as np
import scipy
from pygrex.models.recommender_model import RecommenderModel


class MFImplicitModel(RecommenderModel):
    def __init__(self, latent_dim, reg_term, learning_rate, epochs):
        self.latent_dim = latent_dim
        self.reg_term = reg_term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, dataset):
        self.model.fit(
            self.rearrange_dataset(
                ds=dataset.dataset,
                num_user=dataset.num_user,
                num_item=dataset.num_item,
            )
        )
        return True

    @staticmethod
    def rearrange_dataset(ds, num_user: int, num_item: int) -> scipy.sparse.csr_matrix:
        """
        Converts the dataset into a sparse matrix format for the implicit model.

        Args:
            ds: Dataset containing userId and itemId columns
            num_user : Number of users in the dataset
            num_item : Number of items in the dataset

        Returns:
            ds_mtr: Sparse matrix representation of the dataset
        """

        # Create sparse matrix directly from data
        data = np.ones(len(ds))  # Array of 1s for each interaction
        rows = ds["userId"].values  # User IDs as row indices
        cols = ds["itemId"].values  # Item IDs as column indices

        ds_mtr = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_user, num_item)
        )

        return ds_mtr

    def predict(self, user_id, item_id):
        """
        Predict ratings for a user and one or more items using efficient vectorization.

        Args:
            user_id : User identifier
            item_id : Item identifier or a list/array of item identifiers

        Returns:
            A single predicted score (float) or an array of scores (np.ndarray)
        """
        # 1. Validate user_id
        if not (0 <= user_id < self.model.user_factors.shape[0]):
            raise ValueError(f"user_id {user_id} is out of bounds")

        # 2. Unify input to always be a numpy array
        is_single_item = not isinstance(item_id, (list, np.ndarray))
        item_ids_arr = np.array(item_id, ndmin=1)

        # 3. Perform a single, vectorized bounds check for all items at once
        max_item_id = self.model.item_factors.shape[0]
        if not np.all((item_ids_arr >= 0) & (item_ids_arr < max_item_id)):
            out_of_bounds_id = item_ids_arr[(item_ids_arr < 0) | (item_ids_arr >= max_item_id)][0]
            raise ValueError(f"item_id {out_of_bounds_id} is out of bounds")

        # 4. Get all item vectors in a single, highly efficient operation
        item_vectors = self.model.item_factors[item_ids_arr]
        user_vector = self.model.user_factors[user_id]
        
        # 5. Calculate all scores with one dot product
        scores = user_vector.dot(item_vectors.T)

        # 6. Return a single float if the input was a single item, otherwise the array
        return scores[0] if is_single_item else scores

    def user_embedding(self):
        return self.model.user_factors

    def item_embedding(self):
        return self.model.item_factors
