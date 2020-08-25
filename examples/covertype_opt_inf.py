import argparse
from typing import Text

import tensorflow as tf
import numpy as np

from tabnet.datasets.covertype import get_dataset, get_data
from tabnet.utils import set_seed
from tabnet.models.classify import TabNetClassifier


DATA_PATH = "data/covtype.csv"


def optimize_alpha_on_dataset(model: tf.keras.Model, ds: tf.data.Dataset, size_of_dataset: int):
    sample, _ = next(iter(ds))
    bs = sample.shape[0]
    num_steps = int(np.ceil(size_of_dataset / bs))

    alphas = [1/(2*bs) * i for i in range(50)]

    accuracies = []
    for alpha in alphas:
        accuracy = 0
        ds_iter = iter(ds)
        for _ in range(num_steps):
            x, y_true = next(ds_iter)
            logits = model(x, training=False, alpha=alpha)
            probs = tf.nn.softmax(logits)
            y_pred = tf.argmax(probs, axis=-1)
            accuracy += tf.reduce_sum(tf.cast(y_pred == y_true, tf.float32)).numpy()
        accuracies.append(accuracy / size_of_dataset)

    return alphas, accuracies


def main(model_dir: Text, data_path: Text, batch_size: int, seed: int):
    set_seed(seed)
    model = TabNetClassifier.load_from_directory(model_dir)

    _, _, df_test = get_data(data_path)

    ds_test = get_dataset(
        df_test, shuffle=False, batch_size=batch_size, drop_remainder=False
    )

    alphas, accuracies = optimize_alpha_on_dataset(model, ds_test, len(df_test))

    print(accuracies)

    print(f"Accuracy: {np.min(accuracies)} (min) {np.max(accuracies)} (max)")
    print(f"Alphas: {np.min(alphas)} (min) {np.max(alphas)} (max)")
    print(f"Alpha at Accuracy: {alphas[np.argmin(accuracies)]} (min) {alphas[np.argmax(accuracies)]} (max)")


# python examples/covertype_opt_inf.py --model_dir .outs/test/w200
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--data_path", default=DATA_PATH, type=str)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    main(
        args.model_dir,
        args.data_path,
        args.batch_size,
        args.seed
    )
