import pytest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.mlp_model import MLPModel
from recoxplainer.recommender import Recommender


@pytest.fixture
def setup_data():
    mlp = MLPModel(**cfg.model.mlp)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    data.binarize()
    return mlp, data


def test_train_mlp(setup_data):
    mlp, data = setup_data
    assert mlp.fit(data)
    recommender = Recommender(data, mlp)
    recommender.recommend_all()
