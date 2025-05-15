import pytest

from recoxplainer.config import cfg
from recoxplainer.data_reader.data_reader import DataReader


@pytest.fixture
def data_reader():
    return DataReader(**cfg.data.testdata)


def test_import(data_reader):
    assert data_reader.num_user == 249
    assert data_reader.num_item == 551
    assert data_reader.dataset.shape[0] == 1000
    assert data_reader.dataset.shape[1] == 4
