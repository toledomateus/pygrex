import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.auto import tqdm

from pygrex.data_reader import DataReader, UserItemRatingDataset
from pygrex.utils.torch_utils import use_optimizer
from .py_torch_model import PyTorchModel


class MLPModel(PyTorchModel):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        latent_dim: int,
        epochs: int,
        num_negative: int,
        batch_size: int,
        cuda: bool,
        optimizer_name: str,
        device_id=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            cuda=cuda,
            optimizer_name=optimizer_name,
            device_id=device_id,
        )

        self.negative_sample_size = num_negative
        self.weight_decay = weight_decay

        # layer dim is 2*self.latent_dim since the embeddings will be concatenated
        self.affine_output = torch.nn.Linear(
            in_features=2 * self.latent_dim, out_features=1
        )
        self.logistic = torch.nn.Sigmoid()

        self.criterion = nn.BCELoss()
        self.optimizer: Optimizer | None = None

    def fit(self, data: DataReader):
        optimizer = use_optimizer(
            network=self,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            optimizer_name=self.optimizer_name,
        )
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected an Optimizer, but got {type(optimizer)}")
        self.optimizer = optimizer

        dataset = data.dataset

        num_users = data.num_user
        num_items = data.num_item

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=num_users, embedding_dim=self.latent_dim
        )

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=num_items, embedding_dim=self.latent_dim
        )

        self.negatives = self._sample_negative(dataset)

        with tqdm(total=self.epochs) as progress:
            for epoch in range(self.epochs):
                train_loader = self.instance_a_train_loader(
                    dataset, self.negative_sample_size, self.batch_size
                )
                loss = self.train_an_epoch(train_loader)
                progress.update(1)
                progress.set_postfix({"loss": loss})

    def instance_a_train_loader(self, dataset, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(
            dataset, self.negatives[["userId", "negative_items"]], on="userId"
        )
        train_ratings["negatives"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(list(x), num_negatives)
        )
        user_ids = train_ratings["userId"].tolist()
        item_ids = train_ratings["itemId"].tolist()
        rating_values = train_ratings["rating"].tolist()
        negatives_lists = train_ratings["negatives"].tolist()

        for user, item, rating, negatives in zip(
            user_ids, item_ids, rating_values, negatives_lists
        ):
            users.append(user)
            items.append(item)
            ratings.append(rating)
            for neg_item in negatives:
                users.append(user)
                items.append(neg_item)
                ratings.append(float(0))  # negative samples get 0 rating

        dataset = UserItemRatingDataset(
            user_tensor=torch.LongTensor(users),
            item_tensor=torch.LongTensor(items),
            target_tensor=torch.FloatTensor(ratings),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_an_epoch(self, train_loader):
        self.train()
        cnt = 0
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
            cnt += 1
        return total_loss / cnt

    def train_single_batch(self, users, items, ratings):
        if self.cuda is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer is not initialized. Call fit() before training."
            )

        self.optimizer.zero_grad()
        ratings_pred = self(users, items)
        loss = self.criterion(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = (
            ratings.groupby("userId")["itemId"]
            .apply(set)
            .reset_index()
            .rename(columns={"itemId": "interacted_items"})
        )
        self.item_catalogue = set(ratings.itemId)
        interact_status["negative_items"] = interact_status["interacted_items"].apply(
            lambda x: self.item_catalogue - x
        )
        return interact_status[["userId", "negative_items"]]

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        # concat latent vector

        # this is needed because cat does not support broadcasting.
        if user_embedding.size()[0] == 1:
            user_embedding = user_embedding.repeat(item_embedding.size()[0], 1)

        element_concat = torch.cat((user_embedding, item_embedding), 1)
        concat = self.affine_output(element_concat)
        rating = self.logistic(concat)
        return rating
