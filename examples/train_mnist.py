import argparse
from typing import Text, List
import pickle
import shutil
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from kerastuner.tuners import RandomSearch

from tabnet.models.classify import TabNetClassifier
from tabnet.utils import set_seed
from tabnet.schedules import DecayWithWarmupSchedule


SEARCH_DIR = ".search"
SEED = 42
DEFAULTS = {"num_features": 784, "n_classes": 10, "min_learning_rate": 1e-6}  # 28x28


# because doing a training on MNIST is something I MUST do, no?
# this time let's add a twist & do hyperparameter optimization with kerastuner


def build_model(hp):
    model = TabNetClassifier(
        num_features=DEFAULTS["num_features"],
        feature_dim=hp.Choice("feature_dim", values=[16, 32, 64], default=32),
        output_dim=hp.Choice("output_dim", values=[16, 32, 64], default=32),
        n_classes=DEFAULTS["n_classes"],
        n_step=hp.Choice("n_step", values=[2, 4, 5, 6], default=4),
        relaxation_factor=hp.Choice(
            "relaxation_factor", values=[1.0, 1.25, 1.5, 2.0, 3.0], default=1.5
        ),
        sparsity_coefficient=hp.Choice(
            "sparsity_coefficient", values=[0.0001, 0.001, 0.01, 0.02, 0.05], default=0.0001
        ),
        bn_momentum=hp.Choice("bn_momentum", values=[0.6, 0.7, 0.9], default=0.7),
        bn_virtual_divider=1,  # let's not use Ghost Batch Normalization. batch sizes are too small
        dp=hp.Choice("dp", values=[0.0, 0.1, 0.2, 0.3, 0.4], default=0.0)
    )
    lr = DecayWithWarmupSchedule(
        hp.Choice("learning_rate", values=[0.001, 0.005, 0.01, 0.02, 0.05], default=0.02),
        DEFAULTS["min_learning_rate"],
        hp.Choice("warmup", values=[1, 5, 10, 20], default=5),
        hp.Choice("decay_rate", values=[0.8, 0.90, 0.95, 0.99], default=0.95),
        hp.Choice("decay_steps", values=[10, 100, 500, 1000], default=500),
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        clipnorm=hp.Choice("clipnorm", values=[1, 2, 5, 10], default=2),
    )

    lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer,
        loss=lossf,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def prepare_dataset(
    ds: tf.data.Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_remainder: bool = False,
):
    size_of_dataset = ds.reduce(0, lambda x, _: x + 1).numpy()
    if shuffle:
        ds = ds.shuffle(buffer_size=size_of_dataset, seed=SEED)
    ds: tf.data.Dataset = ds.batch(batch_size, drop_remainder=drop_remainder)

    @tf.function
    def prepare_data(features):
        image = tf.cast(features["image"], tf.float32)
        bs = tf.shape(image)[0]
        image = tf.reshape(image / 255.0, (bs, -1))
        return image, features["label"]

    autotune = tf.data.experimental.AUTOTUNE
    ds = ds.map(prepare_data, num_parallel_calls=autotune).prefetch(autotune)
    return ds


def search(
    epochs: int,
    batch_size: int,
    n_trials: int,
    execution_per_trial: int,
    project: Text,
    do_cleanup: bool,
):
    set_seed(SEED)

    dir_to_clean = os.path.join(SEARCH_DIR, project)
    if do_cleanup and os.path.exists(dir_to_clean):
        shutil.rmtree(dir_to_clean)

    # first 80% for train. remaining 20% for val & test dataset for final eval.
    ds_tr, ds_val, ds_test = tfds.load(
        name="mnist",
        split=["train[:80%]", "train[-20%:]", "test"],
        data_dir="mnist",
        shuffle_files=False,
    )

    ds_tr = prepare_dataset(ds_tr, batch_size, shuffle=True, drop_remainder=True)
    ds_val = prepare_dataset(ds_val, batch_size, shuffle=False, drop_remainder=False)
    ds_test = prepare_dataset(ds_test, batch_size, shuffle=False, drop_remainder=False)

    tuner = RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=n_trials,
        executions_per_trial=execution_per_trial,
        directory=SEARCH_DIR,
        project_name=project,
    )

    # ? add callbacks
    tuner.search(
        ds_tr, epochs=epochs, validation_data=ds_val,
    )

    best_model: tf.keras.Model = tuner.get_best_models(num_models=1)[0]
    best_model.build((None, DEFAULTS["num_features"]))
    results = best_model.evaluate(ds_test, return_dict=True)

    tuner.results_summary(num_trials=1)
    best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)
    print(f"Test results: {results}")

    output = {"results": results, "best_hyperparams": best_hyperparams}
    with open("search_results.pickle", "wb") as f:
        pickle.dump(output, f)


# python3 examples/train_mnist.py --trials 2 --epochs 10 --bs 128
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--exec_per_trial", default=2, type=int)
    parser.add_argument("--project", default="test", type=str)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    search(
        args.epochs,
        args.bs,
        args.trials,
        args.exec_per_trial,
        args.project,
        args.cleanup,
    )
