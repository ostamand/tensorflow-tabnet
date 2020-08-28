import pytest
import tensorflow as tf

from tabnet.models.gbn import BatchNormInferenceWeighting


class TestCustomBatchNorm(tf.test.TestCase):
    def setUp(self):
        self.x = tf.random.uniform(shape=(32, 54), dtype=tf.float32)
        self.zeros = tf.zeros(self.x.shape[1])
        self.ones = tf.ones(self.x.shape[1])

    def test_can_apply_bn_in_training(self):
        bn = BatchNormInferenceWeighting()
        x_bn = bn(self.x, training=True)

        mean = tf.reduce_mean(x_bn, axis=0)
        std = tf.sqrt(
            tf.reduce_mean(tf.pow(x_bn, 2), axis=0)
            - tf.pow(tf.reduce_mean(x_bn, axis=0), 2)
        )

        self.assertAllClose(mean, self.zeros, rtol=1e-04, atol=1e-04)
        self.assertAllClose(std, self.ones, rtol=1e-04, atol=1e-04)

    def test_update_moving_stats_only_in_training(self):
        bn = BatchNormInferenceWeighting()
        _ = bn(self.x, training=False)

        self.assertAllClose(bn.moving_mean, self.zeros)
        self.assertAllClose(bn.moving_mean_of_squares, self.zeros)

        _ = bn(self.x, training=True)

        self.assertNotAllClose(bn.moving_mean, self.zeros)
        self.assertNotAllClose(bn.moving_mean_of_squares, self.zeros)

    def test_similar_to_keras(self):
        bn = BatchNormInferenceWeighting(momentum=0.9)
        bn_keras = tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=tf.keras.backend.epsilon()
        )

        x_bn = bn(self.x, training=True)
        x_bn_keras = bn_keras(self.x, training=True)

        self.assertAllClose(x_bn, x_bn_keras, rtol=1e-4, atol=1e-4)
        # TODO check moving mean & std


if __name__ == "__main__":
    tf.test.main()
