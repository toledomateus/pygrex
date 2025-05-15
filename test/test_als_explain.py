import pytest
from recoxplainer.config import cfg
from recoxplainer.data_reader import DataReader
from recoxplainer.models import ALS
from recoxplainer.recommender import Recommender
from recoxplainer.evaluator import Evaluator, Splitter
from recoxplainer.explain import KNNPostHocExplainer
from threadpoolctl import threadpool_limits


# Fixture to replace setUp method
@pytest.fixture(autouse=True)
def limit_blas_threads():
    threadpool_limits(1, "blas")


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

    # Test ALS model fitting
    assert als.fit(train)

    # Generate recommendations
    recommender = Recommender(data, als)
    recommendations = recommender.recommend_all()

    # Evaluate recommendations
    evaluator = Evaluator(test)
    evaluator.cal_hit_ratio(recommendations)

    # explainer = ALSExplainer(als, recommendations, data)
    # explainer.explain_recommendations()

    # KNN Post Hoc Explainer
    knn_explainer = KNNPostHocExplainer(als, recommendations, train)
    knn_explainer.explain_recommendations()
