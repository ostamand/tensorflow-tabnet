import os
import shutil
from argparse import ArgumentParser
from typing import Text
from datetime import datetime

import tensorflow as tf
import numpy as np
from kerastuner.tuners import BayesianOptimization, RandomSearch

from tabnet.models import TabNetClassifier
from local.original_dataset import (
    input_fn,
    get_columns,
    NUM_FEATURES,
    N_VAL_SAMPLES,
    N_TR_SAMPLES,
)


SEARCH_DIR = ".search"


DEFAULTS = {
    "N_d": 64,
    "N_a": 64,
    "sparsity_coefficient": 0.0001,
    "batch_size": 16384,
    "bn_virtual_bs": 512,  # 256
    "bn_momentum": 0.7,
    "n_steps": 5,
    "relaxation_factor": 1.5,
    "n_classes": 7,
    "learning_rate": 0.02,
    "decay_steps": 500,  # 20
    "decay_rate": 0.95,
    "total_steps": 130000,
    "gradient_thresh": 2000,
}


def build_model(hp):
    model = TabNetClassifier(
        feature_columns=get_columns(),
        num_features=NUM_FEATURES,
        feature_dim=DEFAULTS["N_d"],
        output_dim=DEFAULTS["N_a"],
        n_classes=DEFAULTS["n_classes"],
        n_step=DEFAULTS["n_steps"],
        relaxation_factor=DEFAULTS["relaxation_factor"],
        sparsity_coefficient=hp.Choice(
            "sparsity_coefficient", values=[0.0001, 0.001, 0.01], default=0.0001
        ),
        bn_momentum=hp.Choice("bn_momentum", values=[0.6, 0.7, 0.9], default=0.7),
        bn_virtual_bs=hp.Choice(
            "bn_virtual_bs", values=[128, 256, 512, 1024], default=512
        ),
    )

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        hp.Choice(
            "learning_rate", values=[0.002, 0.005, 0.01, 0.02, 0.05], default=0.02
        ),
        decay_steps=hp.Choice("decay_steps", values=[10, 100, 500, 1000], default=500),
        decay_rate=hp.Choice("decay_rate", values=[0.90, 0.95, 0.99], default=0.95),
        staircase=False,
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


# Total runtime: 158.35 mins = 10 trials
def search(
    epochs: int,
    n_trials: int,
    execution_per_trial: int,
    project: Text = "test",
    cleanup: bool = False,
):
    start_time = datetime.now()

    results_path = os.path.join(SEARCH_DIR, project)
    if cleanup and os.path.exists(results_path):
        shutil.rmtree(results_path)

    ds_tr = input_fn(
        "data/train_covertype.csv", shuffle=True, batch_size=DEFAULTS["batch_size"]
    )
    ds_val = input_fn(
        "data/val_covertype.csv", shuffle=False, batch_size=DEFAULTS["batch_size"]
    )

    num_train_steps = np.floor(N_TR_SAMPLES / DEFAULTS["batch_size"])
    num_valid_steps = np.floor(N_VAL_SAMPLES / DEFAULTS["batch_size"])

    # RandomSearch, BayesianOptimization
    tuner = RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=n_trials,
        executions_per_trial=execution_per_trial,
        directory=SEARCH_DIR,
        project_name=project,
    )

    # tuner.search_space_summary()

    tuner.search(
        ds_tr,
        epochs=epochs,
        validation_data=ds_val,
        steps_per_epoch=num_train_steps,
        validation_steps=num_valid_steps,
    )

    # models = tuner.get_best_models(num_models=1)

    tuner.results_summary(num_trials=2)

    print(f"Total runtime: {(datetime.now() - start_time).seconds / 60:.2f} mins")


# python local/tuner_orig_dataset.py --trials 10 --epoch 25 --project test_rnd_10_trials
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trials", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--exec_per_trial", default=2, type=int)
    parser.add_argument("--project", default="test", type=str)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    search(args.epochs, args.trials, args.exec_per_trial, args.project, args.cleanup)
