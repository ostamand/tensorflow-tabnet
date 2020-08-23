import pytest
import tensorflow as tf
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes

from tabnet.models import TabNet
from tabnet.models.transformers import FeatureBlock
from tabnet.datasets.covertype import get_data, get_dataset


COVTYPE_CSV_PATH = "data/covtype.csv"
FEATURE_DIM = 50


class TabNetTest(tf.test.TestCase):
    def setUp(self):
        df_tr, _, _ = get_data(COVTYPE_CSV_PATH)
        self.dataset = get_dataset(df_tr)
        self.x, _ = next(iter(self.dataset))
        self.feature_dim = FEATURE_DIM
        self.features = tf.random.uniform([32, FEATURE_DIM], -1.0, 1.0)

    def test_feature_transformer_block(self):
        block = FeatureBlock(self.feature_dim, apply_glu=True, bn_virtual_divider=1)
        x = block(self.features, training=False)
        assert x.shape[1] == self.features.shape[1]

    def test_tabnet_model(self):
        model = TabNet(self.x.shape[1], feature_dim=16, output_dim=16, n_step=2)
        y, _, _ = model(self.x, training=True)
        assert y.shape[0] == self.x.shape[0]
        assert y.shape[1] == 16

    def test_tabnet_with_alpha(self):
        # in training mode alpha should change nothing
        model = TabNet(self.x.shape[1], feature_dim=16, output_dim=16, n_step=2)
        y_with_alpha, _, _ = model(self.x, training=True, alpha=0.5)
        y_no_alpha, _, _ = model(self.x, training=True)

        self.assertAllClose(y_with_alpha, y_no_alpha)

        # in inference mode when alpha > 1.0 the batch stats will be used

        y_with_alpha, _, _ = model(self.x, training=False, alpha=0.5)
        y_no_alpha, _, _ = model(self.x, training=False)

        self.assertNotAllClose(y_with_alpha, y_no_alpha)


# tf.test.main()
