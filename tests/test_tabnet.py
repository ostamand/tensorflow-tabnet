import os
import shutil
import pytest
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes

from tabnet.models import TabNet
from tabnet.models.transformers import FeatureBlock
from tabnet.datasets.covertype import get_data, get_dataset


COVTYPE_CSV_PATH = "data/test/covtype_sample.csv"
FEATURE_DIM = 50
TMP_DIR = ".tmp"


@pytest.fixture()
def features():
    return tf.random.uniform([32, FEATURE_DIM], -1.0, 1.0)


@pytest.fixture()
def model(features):
    model = TabNet(features.shape[1], feature_dim=16, output_dim=16, n_step=2)
    model.build(features.shape)
    return model


@pytest.fixture()
def saved_model_path(model: tf.keras.Model):
    path = os.path.join(TMP_DIR, "saved_model")
    model.save_weights(path, overwrite=True)
    yield path
    shutil.rmtree(TMP_DIR)


class TestTabNet:
    def test_feature_transformer_block(self, features):
        block = FeatureBlock(FEATURE_DIM, apply_glu=True, bn_virtual_divider=1)
        x = block(features, training=False)
        assert x.shape[1] == features.shape[1]

    def test_tabnet_model(self, model, features):
        y, _, _ = model(features, training=True)
        assert y.shape[0] == features.shape[0]
        assert y.shape[1] == 16

    def test_tabnet_with_alpha(self, model, features):
        # in training mode alpha should change nothing
        y_with_alpha, _, _ = model(features, training=True, alpha=0.5)
        y_no_alpha, _, _ = model(features, training=True)

        np.allclose(y_with_alpha, y_no_alpha)

        # in inference mode when alpha > 1.0 the batch stats will be used

        y_with_alpha, _, _ = model(features, training=False, alpha=0.5)
        y_no_alpha, _, _ = model(features, training=False)

        np.allclose(y_with_alpha, y_no_alpha)

    # @pytest.skip(msg="Saving takes too much time.")
    def test_can_infer_with_saved_model(
        self, model: tf.keras.Model, features, saved_model_path
    ):
        model.load_weights(saved_model_path)
        out1, _, _ = model(features, training=False, alpha=0.5)
        # out2 will all be zeros since bn moving stats are still zeros at that point
        out2, _, _ = model(features, training=False)
        assert not np.allclose(out1.numpy(), out2.numpy())
