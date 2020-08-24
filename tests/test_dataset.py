import numpy as np
import tensorflow as tf

from tabnet.datasets.covertype import get_data, get_dataset


COVTYPE_CSV_PATH = "data/test/covtype_sample.csv"
SEED = 42


class TestDataset(tf.test.TestCase):
    def test_gets_always_the_same_data(self):
        df_tr, df_val, df_test = get_data(COVTYPE_CSV_PATH, seed=SEED)
        df2_tr, df2_val, df2_test = get_data(COVTYPE_CSV_PATH, seed=SEED)

        self.assertAllClose(
            df_tr.values.astype(np.float32), df2_tr.values.astype(np.float32)
        )
        self.assertAllClose(
            df_val.values.astype(np.float32), df2_val.values.astype(np.float32)
        )
        self.assertAllClose(
            df_test.values.astype(np.float32), df2_test.values.astype(np.float32)
        )

    def __get_labels(self, ds: tf.data.Dataset, n_iter: int):
        labels = []
        ds_iter = iter(ds)
        for i in range(n_iter):
            _, label = next(ds_iter)
            labels.append(label)
        return tf.concat(labels, axis=0)

    def test_dataset_is_deterministic(self):
        df_tr, _, _ = get_data(COVTYPE_CSV_PATH, seed=SEED)

        ds_tr = get_dataset(df_tr, shuffle=True, batch_size=32, seed=SEED, take=2)
        labels1 = self.__get_labels(ds_tr, 20)

        ds_tr = get_dataset(df_tr, shuffle=True, batch_size=32, seed=SEED, take=2)
        labels2 = self.__get_labels(ds_tr, 20)

        self.assertAllClose(labels1, labels2)


if __name__ == "__main__":
    tf.test.main()
