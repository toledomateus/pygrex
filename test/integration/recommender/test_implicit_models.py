import pytest

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.models.als_model import ALS
from pygrex.models.bpr_model import BPR
from pygrex.recommender import Recommender


@pytest.fixture
def setup_data():
    als = ALS(**cfg.model.als)
    bpr = BPR(**cfg.model.bpr)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    data.binarize()
    return als, bpr, data


def test_train_als(setup_data):
    als, _, data = setup_data
    als.fit(data)
    recommender = Recommender(data, als)
    max_valid = als.model.item_factors.shape[0]
    item_pool = list(range(min(max_valid, 50)))
    _ = recommender.recommend(user_id=0, target_item_id=item_pool)


def test_train_bpr(setup_data):
    _, bpr, data = setup_data
    bpr.fit(data)
    recommender = Recommender(data, bpr)
    max_valid = bpr.model.item_factors.shape[0]
    item_pool = list(range(min(max_valid, 50)))
    _ = recommender.recommend(user_id=0, target_item_id=item_pool)
