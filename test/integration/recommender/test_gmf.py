import pytest

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.models.gmf_model import GMFModel
from pygrex.recommender import Recommender


@pytest.fixture
def setup_data():
    gmf = GMFModel(**cfg.model.gmf)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    data.binarize()
    return gmf, data


def test_train_gmf(setup_data):
    gmf, data = setup_data
    assert gmf.fit(data)
    recommender = Recommender(data, gmf)
    recommender.recommend_all()
