import itertools
from typing import Union
import torch

from pygrex.utils.torch_utils import use_cuda
from .recommender_model import RecommenderModel
from pygrex.data_reader import DataReader


class PyTorchModel(RecommenderModel, torch.nn.Module):
    """Meta Learner

    Note: Subclass should implement self.model !
    """

    def __init__(
        self,
        learning_rate: float,
        latent_dim: int,
        epochs: int,
        batch_size: int,
        cuda: bool,
        optimizer_name: str,
        device_id: Union[int, None] = None,
    ):
        if optimizer_name not in ["sgd", "adam", "rmsprop"]:
            raise Exception("Wrong optimizer.")

        if cuda is True and device_id is not None:
            use_cuda(True, device_id)

        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self._cuda = cuda
        self.optimizer_name = optimizer_name

        self.dataset = None
        self.dataset_metadata = None
        self.embedding_user = None
        self.embedding_item = None
        self.optimizer = None

        super().__init__()

    def fit(self, data: DataReader):
        pass

    def predict(self, user_id, item_id) -> list:
        if isinstance(user_id, int):
            user_id = [user_id]
        if isinstance(item_id, int):
            item_id = [item_id]
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor(item_id)
        with torch.no_grad():
            if self._cuda:
                user_id = user_id.cuda()
                item_id = item_id.cuda()
            pred = self.forward(user_id, item_id).cpu().tolist()
            pred = list(itertools.chain.from_iterable(pred))
            return pred

    def user_embedding(self):
        return self.state_dict()["embedding_user.weight"].cpu().numpy()

    def item_embedding(self):
        return self.state_dict()["embedding_item.weight"].cpu().numpy()
