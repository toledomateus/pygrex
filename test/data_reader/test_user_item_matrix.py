import pytest
import torch

from pygrex.config import cfg
from pygrex.data_reader.data_reader import DataReader
from pygrex.data_reader.user_item_dict import UserItemDict


@pytest.fixture
def setup_data():
    data = DataReader(**cfg.data.testdata)
    data.make_consecutive_ids_in_dataset()
    return data


def test_user_item_matrix(setup_data):
    data = setup_data

    # Get dimensions for expl_matrix
    n_users = data.dataset.userId.nunique()
    n_items = data.dataset.itemId.nunique()

    # Create a zero-filled expl_matrix
    expl_matrix = torch.zeros((n_users, n_items))

    # Set expl to False for this test
    expl = False

    # Initialize UserItemDict with all required parameters
    user_dict = UserItemDict(data.dataset, expl_matrix, expl)

    # Test that the first rating is correctly stored
    x = data.dataset.userId[0]
    y = data.dataset.itemId[0]
    v = data.dataset.rating[0]
    assert user_dict[x][y] == v
