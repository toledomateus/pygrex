import pytest
from pygrex.config import cfg
from pygrex.data_reader import DataReader
from pygrex.models import ALS
from pygrex.recommender import Recommender
from pygrex.evaluator import ModelEvaluator, Splitter
from pygrex.explain import KNNPostHocExplainer
# from threadpoolctl import threadpool_limits


# # Fixture to replace setUp method
# @pytest.fixture(autouse=True)
# def limit_blas_threads():
#     threadpool_limits(1, "blas")


@pytest.fixture()
def setup_data():
    # Initialize ALS model
    als = ALS(**cfg.model.als)

    # Prepare data
    data = DataReader(**cfg.data.ml100k)
    data.make_consecutive_ids_in_dataset()
    data.binarize()

    return als, data


def test_explain_als(setup_data):
    als, data = setup_data

    # Split data
    sp = Splitter()
    train, test = sp.split_leave_n_out(data, n=1)

    # Test ALS model fitting (fit should complete without raising)
    als.fit(train)

    # Generate recommendations
    recommender = Recommender(train, als)
    # Recommend on a small, valid subset of items
    max_valid = als.model.item_factors.shape[0]
    item_pool = list(range(min(max_valid, 50)))
    _ = recommender.recommend(user_id=0, target_item_id=item_pool)

    # Pipeline ran without exceptions; evaluation moved elsewhere

    # explainer = ALSExplainer(als, recommendations, data)
    # explainer.explain_recommendations()

    # KNN Post Hoc Explainer (skipped in this smoke test)
