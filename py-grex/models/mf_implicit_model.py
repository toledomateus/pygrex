import numpy as np
import scipy


class MFImplicitModel:
    def __init__(self, latent_dim, reg_term, learning_rate, epochs):
        self.latent_dim = latent_dim
        self.reg_term = reg_term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, dataset):
        self.model.fit(
            self.rearrange_dataset(
                ds=dataset.dataset, num_user=dataset.num_user, num_item=dataset.num_item
            )
        )
        return True

    @staticmethod
    def rearrange_dataset(ds, num_user, num_item):
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
        Predict ratings for a user and one or more items.

        Parameters:
        -----------
        user_id : int
            User identifier
        item_id : int or list
            Item identifier or list of item identifiers

        Returns:
        --------
        float or numpy.ndarray
            Predicted rating(s)
        """
        if not (0 <= user_id < self.model.user_factors.shape[0]):
            raise ValueError(f"user_id {user_id} out of bounds")

        # Handle both single item_id and list of item_ids
        if isinstance(item_id, list):
            # Check bounds for all items
            for iid in item_id:
                if not (0 <= iid < self.model.item_factors.shape[0]):
                    raise ValueError(f"item_id {iid} out of bounds")

            # Get user factors
            user_vec = self.model.user_factors[user_id]

            # Get item factors for all items in the list
            item_vecs = np.array([self.model.item_factors[iid] for iid in item_id])

            # Calculate predictions for all items
            return np.dot(user_vec, item_vecs.T)
        else:
            # Original code for single item
            if not (0 <= item_id < self.model.item_factors.shape[0]):
                raise ValueError(f"item_id {item_id} out of bounds")
            return np.dot(
                self.model.user_factors[user_id], self.model.item_factors[item_id]
            )

    def user_embedding(self):
        return self.model.user_factors

    def item_embedding(self):
        return self.model.item_factors
