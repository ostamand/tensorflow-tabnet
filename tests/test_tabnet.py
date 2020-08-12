import pytest
import tensorflow as tf
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes

from tabnet.models import TabNet
from tabnet.models.transformers import FeatureTransformerBlock
from tabnet.datasets.covertype import get_data, build_dataset


FEATURE_DIM = 50
COVTYPE_CSV_PATH = "data/covtype.csv"


@pytest.fixture
def features():
    return tf.random.uniform([32, FEATURE_DIM], -1.0, 1.0)


@pytest.fixture
def dataset():
    df_tr, _, _ = get_data(COVTYPE_CSV_PATH)
    return build_dataset(df_tr)


@run_all_in_graph_and_eager_modes
def test_feature_transformer_block(features):
    block = FeatureTransformerBlock(FEATURE_DIM, apply_glu=True)
    x = block(features)
    assert x.shape[1] == features.shape[1]


@run_all_in_graph_and_eager_modes
def test_tabnet_model(dataset):
    x, _ = next(iter(dataset))
    model = TabNet(x.shape[1], feature_dim=16, output_dim=16, n_step=2)
    y = model(x, training=True)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == 16


if __name__ == "__main__":
    pytest.main(__file__)
