import argparse
import os
from typing import Text
from datetime import datetime
import json

import tensorflow as tf
import numpy as np

from tabnet.models.classify import TabNetClassifier
from tabnet.datasets.covertype import get_dataset, get_data


LOGDIR = ".logs"
OUTDIR = ".outs"
DATA_PATH = "data/covtype.csv"
CONFIGS = {
    "feature_dim": 64,
    "output_dim": 64,
    "num_features": 54,
    "sparsity_coefficient": 0.0001,
    "batch_size": 16384,
    "bn_virtual_bs": 512,
    "bn_momentum": 0.7,
    "n_steps": 5,
    "relaxation_factor": 1.5,
    "n_classes": 7,
    "learning_rate": 0.02,
    "decay_steps": 500,
    "decay_rate": 0.95,
    "total_steps": 130000,
    "clipnorm": 2.0,
    "patience": 200,
}


def train(
    run_name: Text,
    data_path: Text,
    out_dir: Text,
    bn_momentum: float,
    bn_virtual_bs: int,
    clipnorm: float,
    decay_rate: float,
    decay_steps: int,
    learning_rate: float,
    sparsity_coefficient: float,
    epochs: int
):
    df_tr, df_val, df_test = get_data(data_path)

    ds_tr = get_dataset(df_tr, shuffle=True, batch_size=CONFIGS["batch_size"])
    # TODO check drop_remainder
    ds_val = get_dataset(df_val, shuffle=False, batch_size=CONFIGS["batch_size"])
    ds_test = get_dataset(df_test, shuffle=False, batch_size=CONFIGS["batch_size"])

    num_train_steps = np.floor(len(df_tr) / CONFIGS["batch_size"])
    num_valid_steps = np.floor(len(df_val) / CONFIGS["batch_size"])
    num_test_steps = np.floor(len(df_test) / CONFIGS["batch_size"])

    model = TabNetClassifier(
        num_features=CONFIGS["num_features"],
        feature_dim=CONFIGS["feature_dim"],
        output_dim=CONFIGS["output_dim"],
        n_classes=CONFIGS["n_classes"],
        n_step=CONFIGS["n_steps"],
        relaxation_factor=CONFIGS["relaxation_factor"],
        sparsity_coefficient=sparsity_coefficient,
        bn_momentum=bn_momentum,
        bn_virtual_bs=bn_virtual_bs,
    )

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, clipnorm=clipnorm
    )

    lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer,
        loss=lossf,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    epochs = int(np.ceil(CONFIGS["total_steps"] / num_train_steps)) if epochs is None else epochs

    log_dir = (
        os.path.join(LOGDIR, datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
        if run_name is None
        else os.path.join(LOGDIR, run_name)
    )

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, write_graph=True, profile_batch=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=CONFIGS["patience"],
            verbose=1,
            mode="max",
            restore_best_weights=True,
        ),
    ]

    # train

    h = model.fit(
        ds_tr,
        epochs=epochs,
        validation_data=ds_val,
        steps_per_epoch=num_train_steps,
        validation_steps=num_valid_steps,
        callbacks=callbacks,
    )

    # evaluate

    metrics = model.evaluate(ds_test, steps=num_test_steps, return_dict=True)
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(metrics, f)

    model.save(os.path.join(OUTDIR, "tabnet"))

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TabNet Covertype Training")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--data_path", default=DATA_PATH, type=str)
    parser.add_argument("--out_dir", default=OUTDIR, type=str)
    parser.add_argument("--bn_momentum", default=CONFIGS["bn_momentum"], type=float)
    parser.add_argument("--bn_virtual_bs", default=CONFIGS["bn_virtual_bs"], type=int)
    parser.add_argument("--clipnorm", default=CONFIGS["clipnorm"], type=float)
    parser.add_argument("--decay_rate", default=CONFIGS["decay_rate"], type=float)
    parser.add_argument("--decay_steps", default=CONFIGS["decay_steps"], type=int)
    parser.add_argument("--learning_rate", default=CONFIGS["learning_rate"], type=int)
    parser.add_argument("--sparsity_coefficient", default=CONFIGS["sparsity_coefficient"], type=float)
    parser.add_argument("--epochs", default=None, type=int)
    args = parser.parse_args()

    train(
        args.run_name,
        args.data_path,
        args.out_dir,
        args.bn_momentum,
        args.bn_virtual_bs,
        args.clipnorm,
        args.decay_rate,
        args.decay_steps,
        args.learning_rate,
        args.sparsity_coefficient,
        args.epochs
    )
