import pytest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models import EMFModel
from recoxplainer.recommender import Recommender


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
