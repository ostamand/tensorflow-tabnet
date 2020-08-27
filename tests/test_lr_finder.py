import os
import pytest

import numpy as np
import tensorflow as tf

from tabnet.callbacks.lrfinder import LRFinder


BS = 16
NUM_STEPS = 10
FIGNAME = "test.png"


@pytest.fixture()
def dataset() -> tf.data.Dataset:
    # generate fake data
    size_of_dataset = BS * 10

    x = tf.random.uniform((size_of_dataset,1), minval=-1, maxval=1)
    y = (2 * tf.random.uniform((size_of_dataset,1), minval=0.9, maxval=1.1)) * x + 10 + tf.random.uniform((size_of_dataset,1), minval=-2., maxval=2.)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(x.shape[0]).batch(BS, drop_remainder=False)
    return ds


@pytest.fixture()
def model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(1)]
    )


@pytest.fixture
def clean_output():
    yield
    os.remove(FIGNAME)


@pytest.mark.usefixtures("clean_output")
def test_can_run_the_lr_finder(model: tf.keras.Model, dataset: tf.data.Dataset):
    min_lr = 1e-6
    max_lr = 1e-1

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.MeanSquaredError()
    )

    lrfinder = LRFinder(min_lr, max_lr, num_steps=NUM_STEPS, figname=FIGNAME)

    model.fit(
        dataset,
        epochs=1,
        callbacks=[
            lrfinder
        ]
    )

    assert len(lrfinder.losses) == NUM_STEPS
    assert len(lrfinder.lrs) == NUM_STEPS
    assert lrfinder.lrs[0] == min_lr
    assert lrfinder.lrs[-1] == max_lr

    # by default should have saved a figure with the results
    assert os.path.exists(lrfinder.figname)

