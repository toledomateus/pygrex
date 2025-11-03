import numpy as np
import scipy
from typing import Union, Protocol, runtime_checkable

from implicit.recommender_base import RecommenderBase
from .recommender_model import RecommenderModel
from pygrex.data_reader import DataReader


@runtime_checkable
class FittableImplicitModel(Protocol):
    user_factors: np.ndarray
    item_factors: np.ndarray

    def fit(self, item_user_data) -> None: ...


class MFImplicitModel(RecommenderModel):
    def __init__(
        self,
        latent_dim,
        reg_term,
        learning_rate,
        epochs,
        num_users=None,
        num_items=None,
    ):
        self.latent_dim = latent_dim
        self.reg_term = reg_term
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model: Union[RecommenderBase, FittableImplicitModel, None] = None
        self.total_users = num_users
        self.total_items = num_items

    def fit(self, data: DataReader) -> None:
        if self.model is None:
            raise RuntimeError(
                "The model has not been initialized. Please use a specific subclass like ALS or BPR."
            )
        num_user_for_shape = data.dataset["userId"].max() + 1
        num_item_for_shape = data.dataset["itemId"].max() + 1
        self.total_users = num_user_for_shape
        self.total_items = num_item_for_shape

        item_user_data = self.rearrange_dataset(
            ds=data.dataset,
            num_user=num_user_for_shape,
            num_item=num_item_for_shape,
        ).T.tocsr()

        self.model.fit(item_user_data)

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

    def predict(
        self, user_id: Union[str, int], item_id: Union[str, int, list, np.ndarray]
    ) -> Union[float, list]:
        """
        Predict ratings for a user and one or more items using efficient vectorization.

        Args:
            user_id : User identifier
            item_id : Item identifier or a list/array of item identifiers

        Returns:
            A single predicted score (float) or an array of scores (np.ndarray)
        """
        if not isinstance(self.model, FittableImplicitModel):
            raise RuntimeError(
                "The model has not been trained yet. Please call fit() first."
            )
        user_id = int(user_id)

        # 1. Validate user_id
        if not (0 <= user_id < self.model.user_factors.shape[0]):
            raise ValueError(f"user_id {user_id} is out of bounds")

        # 2. Unify input to always be a numpy array
        is_single_item = not isinstance(item_id, (list, np.ndarray))
        item_ids_arr = np.array(item_id, ndmin=1).astype(int)

        # 3. Perform a single, vectorized bounds check for all items at once
        max_item_id = self.model.item_factors.shape[0]
        if not np.all((item_ids_arr >= 0) & (item_ids_arr < max_item_id)):
            out_of_bounds_id = item_ids_arr[
                (item_ids_arr < 0) | (item_ids_arr >= max_item_id)
            ][0]
            raise ValueError(f"item_id {out_of_bounds_id} is out of bounds")

        # 4. Get all item vectors in a single, highly efficient operation
        item_vectors = self.model.item_factors[item_ids_arr]
        user_vector = self.model.user_factors[user_id]

        # 5. Calculate all scores with one dot product
        scores = user_vector.dot(item_vectors.T)

        # 6. Return a single float if the input was a single item, otherwise the array
        return scores[0].item() if is_single_item else scores.tolist()

    def user_embedding(self) -> np.ndarray:
        if not isinstance(self.model, FittableImplicitModel):
            raise RuntimeError(
                "The model has not been trained yet. Please call fit() first."
            )
        return self.model.user_factors

    def item_embedding(self) -> np.ndarray:
        if not isinstance(self.model, FittableImplicitModel):
            raise RuntimeError(
                "The model has not been trained yet. Please call fit() first."
            )
        return self.model.item_factors
