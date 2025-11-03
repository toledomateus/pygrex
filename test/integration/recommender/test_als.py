import pytest

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.models.als_model import ALS
from pygrex.recommender import Recommender


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
    max_valid = als.model.item_factors.shape[0]
    item_pool = list(range(min(max_valid, 50)))
    _ = recommender.recommend(user_id=0, target_item_id=item_pool)
