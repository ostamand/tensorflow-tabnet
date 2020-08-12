import tensorflow as tf

from tabnet.models.utils import glu


class FeatureTransformerBlock(tf.keras.Model):
    def __init__(
        self,
        feature_dim: int,
        bn_momentum: float,
        bn_virtual_bs: int,
        apply_glu: bool = True,
    ):
        """Feature Transformer Block

        TODO:
            - BN momentum
            - Virtual batch size
        """
        super(FeatureTransformerBlock, self).__init__()
        self.feature_dim = feature_dim
        self.apply_glu = apply_glu

        units = feature_dim * 2 if apply_glu else feature_dim

        self.fc = tf.keras.layers.Dense(units, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, virtual_batch_size=bn_virtual_bs
        )

    def call(self, x, training=None):
        x = self.bn(self.fc(x), training=training)
        if self.apply_glu:
            return glu(x, self.feature_dim)
        return x


class SharedFeatureTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_bs: int):
        super(SharedFeatureTransformer, self).__init__()

        self.block1 = FeatureTransformerBlock(
            feature_dim, bn_momentum=bn_momentum, bn_virtual_bs=bn_virtual_bs
        )
        self.block2 = FeatureTransformerBlock(
            feature_dim, bn_momentum=bn_momentum, bn_virtual_bs=bn_virtual_bs
        )

    def call(self, x, training=None):
        x1 = self.block1(x, training=training)
        x2 = self.block2(x1, training=training)
        return tf.sqrt(0.5) * x1 + x2


class FeatureTransformer(tf.keras.Model):
    def __init__(
        self,
        shared: SharedFeatureTransformer,
        feature_dim: int,
        bn_momentum: float,
        bn_virtual_bs: int,
    ):
        super(FeatureTransformer, self).__init__()
        self.shared = shared
        self.block1 = FeatureTransformerBlock(
            feature_dim, bn_momentum=bn_momentum, bn_virtual_bs=bn_virtual_bs
        )
        self.block2 = FeatureTransformerBlock(
            feature_dim, bn_momentum=bn_momentum, bn_virtual_bs=bn_virtual_bs
        )

    def call(self, x, training=None):
        x = self.shared(x, training=training)
        x1 = self.block1(x, training=training)
        x1 = tf.sqrt(0.5) * x + x1
        x2 = self.block2(x1, training=training)
        return tf.sqrt(0.5) * x1 + x2


class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_bs: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureTransformerBlock(
            feature_dim,
            bn_momentum=bn_momentum,
            bn_virtual_bs=bn_virtual_bs,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return x * prior_scales
