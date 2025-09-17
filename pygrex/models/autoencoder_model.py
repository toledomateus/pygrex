import numpy as np
import torch
import torch.nn as nn
import torch.optim
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Union, List

from pygrex.utils.torch_utils import use_cuda, use_optimizer
from pygrex.data_reader import UserItemDict, DataReader
from pygrex.models import RecommenderModel


class ExplAutoencoderTorch(RecommenderModel, nn.Module):
    def __init__(
        self,
        hidden_layer_features: int,
        learning_rate: float,
        positive_threshold: float,
        weight_decay: float,
        epochs: int,
        knn: int,
        cuda: bool,
        optimizer_name: str,
        expl: bool,
        device_id: Optional[int] = None,
    ):
        super().__init__()
        if optimizer_name not in ["sgd", "adam", "rmsprop"]:
            raise Exception("Wrong optimizer.")
        if cuda:
            use_cuda(True, device_id if device_id is not None else 0)

        self.positive_threshold = positive_threshold
        self.weight_decay = weight_decay
        self.knn = knn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_gpu = cuda
        self.optimizer_name = optimizer_name
        self.hidden_layer_features = hidden_layer_features
        self.expl = expl

        self.dataset = None
        self.data = None
        self.embedding_user = None
        self.embedding_item = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.explainability_matrix = None
        self.sim_users = {}

        self.criterion = nn.MSELoss()

    def fit(self, data: DataReader):
        self.data = data
        self.dataset = data.dataset
        num_items = self.data.num_item

        self.encoder_hidden_layer = nn.Linear(
            in_features=num_items, out_features=self.hidden_layer_features
        )

        self.decoder_output_layer = nn.Linear(
            in_features=self.hidden_layer_features, out_features=num_items
        )

        self.compute_explainability()
        optimizer = use_optimizer(
            network=self,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            optimizer=self.optimizer_name,
        )

        assert isinstance(optimizer, torch.optim.Optimizer)
        self.optimizer = optimizer

        with tqdm(total=self.epochs) as progress:
            train_loader = self.instance_a_train_loader()
            for epoch in range(self.epochs):
                loss = self.train_an_epoch(train_loader)
                progress.update(1)
                progress.set_postfix({"loss": loss})

    def compute_explainability(self):
        assert self.dataset is not None
        assert self.data is not None
        ds = self.dataset.pivot(index="userId", columns="itemId", values="rating")
        ds = ds.fillna(0)
        ds = sparse.csr_matrix(ds)
        sim_matrix = cosine_similarity(ds)
        min_val = sim_matrix.min() - 1

        for i in range(self.data.num_user):
            sim_matrix[i, i] = min_val

            knn_to_user_i = (-sim_matrix[i, :]).argsort()[: self.knn]
            self.sim_users[i] = knn_to_user_i

        self.explainability_matrix = np.zeros((self.data.num_user, self.data.num_item))

        filter_dataset_on_threshold = self.dataset[
            self.dataset["rating"] >= self.positive_threshold
        ]

        for i in range(self.data.num_user):
            knn_to_user_i = self.sim_users[i]

            rated_items_by_sim_users = filter_dataset_on_threshold[
                filter_dataset_on_threshold["userId"].isin(knn_to_user_i)
            ]

            sim_scores = rated_items_by_sim_users.groupby(by="itemId")
            sim_scores = sim_scores["rating"].sum()
            sim_scores = sim_scores.reset_index()

            self.explainability_matrix[i, sim_scores.itemId] = (
                sim_scores.rating.to_list()
            )

        self.explainability_matrix = MinMaxScaler().fit_transform(
            self.explainability_matrix
        )

        self.explainability_matrix = torch.from_numpy(self.explainability_matrix)

    def instance_a_train_loader(self):
        """instance train loader for one training epoch"""
        assert self.dataset is not None
        assert self.explainability_matrix is not None
        self.user_item_dict = UserItemDict(
            self.dataset, self.explainability_matrix, self.expl
        )
        return DataLoader(self.user_item_dict, shuffle=True)

    def train_an_epoch(self, train_loader):
        self.train()
        cnt = 0
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.Tensor)
            rating = batch[0]
            rating = rating.float()
            loss = self.train_single_user(rating)
            total_loss += loss
            cnt += 1
        return total_loss / cnt

    def train_single_user(self, ratings):
        if self.use_gpu:
            ratings = ratings.cuda()

        assert self.optimizer is not None
        self.optimizer.zero_grad()
        ratings_pred = self(ratings)
        loss = self.criterion(ratings_pred, ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def forward(self, user_adjusted_ratings):
        activation = self.encoder_hidden_layer(user_adjusted_ratings)
        code = torch.relu(activation)
        activation = self.decoder_output_layer(code)
        reconstructed_ratings = torch.relu(activation)
        return reconstructed_ratings

    def predict(
        self, user_id: Union[int, List[int], str], item_id: Union[int, List[int], str]
    ) -> list:
        try:
            if isinstance(user_id, str):
                user_id = int(user_id)
            elif isinstance(user_id, list):
                user_id = [int(u) for u in user_id]
            if isinstance(item_id, str):
                item_id = int(item_id)
            elif isinstance(item_id, list):
                item_id = [int(i) for i in item_id]
        except (ValueError, TypeError):
            raise ValueError(
                "User and item IDs must be integers or strings that can be converted to integers."
            )

        if isinstance(user_id, int):
            user_id = [user_id]
        if isinstance(item_id, int):
            item_id = [item_id]

        with torch.no_grad():
            assert self.user_item_dict is not None, "The model has not been fitted yet."
            rating = self.user_item_dict[user_id]
            rating = rating.float()
            if self.use_gpu:
                rating = rating.cuda()
            pred = self.forward(rating).cpu()
            return pred[:, item_id].tolist()
