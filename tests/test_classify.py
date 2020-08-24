import pytest
import tensorflow as tf
import numpy as np
import shutil

from tabnet.models import TabNetClassifier


CONFIGS = {
    "num_features": 20,
    "feature_dim": 32,
    "output_dim": 64,
    "n_classes": 10
}


OUTPUT_FOLDER = ".tmp"


@pytest.fixture()
def model():
    net = TabNetClassifier(**CONFIGS)
    net.build((None, CONFIGS["num_features"]))
    return net

@pytest.fixture()
def features():
    return tf.random.uniform((32, CONFIGS["num_features"]))*2

@pytest.fixture()
def output_folder():
    yield OUTPUT_FOLDER
    shutil.rmtree(OUTPUT_FOLDER)


class TestClassify():

    def test_can_save_model(self, model, output_folder, features):
        # save to folder
        model.save_to_directory(output_folder)
        out = model(features, training=False, alpha=1)
        # load from folder
        model_loaded = TabNetClassifier.load_from_directory(output_folder)
        out_loaded = model_loaded(features, training=False, alpha=1)

        assert model.configs.keys() == model_loaded.configs.keys()
        for k, v in model_loaded.configs.items():
            assert model.configs[k] == v

        assert np.allclose(model_loaded.head.weights[0].numpy(), model.head.weights[0].numpy())
        assert np.allclose(out, out_loaded)