import pytest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.als_model import ALS
from recoxplainer.recommender import Recommender


@pytest.fixture
def setup_data():
    als = ALS(**cfg.model.als)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    data.binarize()
    als.fit(data)
    return als, data


def test_train_recommend_als(setup_data):
    als, data = setup_data
    recommender = Recommender(data, als)
    recommender.recommend_all()
