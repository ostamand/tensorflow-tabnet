import argparse
import os
from typing import Text
from datetime import datetime
import json

import numpy as np

from tabnet.models.classify import TabNetClassifier
from tabnet.datasets.covertype import build_dataset


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


def train(run_name: Text, data_path: Text, out_dir):
    if tf.config.list_physical_devices("gpu"):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    df_tr, df_val, df_test = get_data(data_path)

    ds_tr = build_dataset(df_tr, shuffle=True, batch_size=CONFIGS["batch_size"])
    ds_val = build_dataset(
        df_val, shuffle=False, batch_size=CONFIGS["batch_size"], drop_remainder=False
    )
    ds_test = build_dataset(
        df_test, shuffle=False, batch_size=CONFIGS["batch_size"], drop_remainder=False
    )

    num_train_steps = np.floor(len(df_tr) / CONFIGS["batch_size"])
    num_valid_steps = np.ceil(len(df_val) / CONFIGS["batch_size"])
    num_test_steps = np.ceil(len(df_test) / CONFIGS["batch_size"])

    with strategy.scope():
        model = TabNetClassifier(
            num_features=CONFIGS["num_features"],
            feature_dim=CONFIGS["N_d"],
            output_dim=CONFIGS["N_a"],
            n_classes=CONFIGS["n_classes"],
            n_step=CONFIGS["n_steps"],
            relaxation_factor=CONFIGS["relaxation_factor"],
            sparsity_coefficient=CONFIGS["sparsity_coefficient"],
            bn_momentum=CONFIGS["bn_momentum"],
            bn_virtual_bs=CONFIGS["bn_virtual_bs"],
        )

        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            CONFIGS["learning_rate"],
            decay_steps=CONFIGS["decay_steps"],
            decay_rate=CONFIGS["decay_rate"],
            staircase=False,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, clipnorm=CONFIGS["clipnorm"]
        )

        lossf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(
            optimizer,
            loss=lossf,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    epochs = int(np.ceil(CONFIGS["total_steps"] / num_train_steps))

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
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(metrics, f)

    model.save(os.path.join(OUTDIR, "best_weights.hdf5"))

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TabNet Covertype Training")
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--out_dir", default=OUTDIR, type=str)
    parser.add_argument("--data_path", default=DATA_PATH, type=str)
    args = parser.parse_args()

    train(args.run_name, args.data_path, args.out_dir)
