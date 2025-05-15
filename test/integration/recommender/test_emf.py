import pytest

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.models import EMFModel
from pygrex.recommender import Recommender


@pytest.fixture
def setup_data():
    emf = EMFModel(**cfg.model.emf)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    return emf, data


def test_train_emf(setup_data):
    emf, data = setup_data
    assert emf.fit(data)
    recommender = Recommender(data, emf)
    recommender.recommend_all()
