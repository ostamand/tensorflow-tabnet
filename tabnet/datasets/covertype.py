from typing import Text, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


RANDOM_SEED = 0


def get_data(path_to_csv: Text, random_seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame]:
    df = pd.read_csv(path_to_csv)
    df_tr, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
    df_tr, df_val = train_test_split(
        df_tr, test_size=0.2 / 0.6, random_state=random_seed
    )
    return df_tr, df_val, df_test


def get_dataset(
    df: pd.DataFrame,
    shuffle: bool = False,
    batch_size: int = 16384,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    x = df[df.columns[:-1]].values.astype(np.float32)
    y = df[df.columns[-1]].values - 1
    ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).repeat()
    return ds
