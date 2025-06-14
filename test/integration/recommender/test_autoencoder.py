import pytest

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.models.autoencoder_model import ExplAutoencoderTorch
from pygrex.recommender import Recommender


@pytest.fixture
def setup_data():
    autoencoder = ExplAutoencoderTorch(**cfg.model.autoencoder)
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    data.binarize()
    return autoencoder, data


def test_train_autoencoder(setup_data):
    autoencoder, data = setup_data
    assert autoencoder.fit(data)
    recommender = Recommender(data, autoencoder)
    recommender.recommend_all()
