import tensorflow as tf


class GhostBatchNormalization(tf.keras.Model):
    def __init__(
        self, virtual_divider: int = 128, momentum: float = 0.9, epsilon: float = 1e-5
    ):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_divider = virtual_divider
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, x, training=None):
        if training:
            chunks = tf.split(x, self.virtual_divider)
            x = [self.bn(x, training=True) for x in chunks]
            return tf.concat(x, 0)
        return self.bn(x, training=False)

    @property
    def moving_mean(self):
        return self.bn.moving_mean

    @property
    def moving_variance(self):
        return self.bn.moving_variance
