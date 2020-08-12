import tensorflow as tf

from tabnet.models.utils import glu


class FeatureTransformerBlock(tf.keras.Model):
    def __init__(self, feature_dim: int, apply_glu: bool = True):
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
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        x = self.bn(self.fc(x))
        if self.apply_glu:
            return glu(x, self.feature_dim)
        return x


class SharedFeatureTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int):
        super(SharedFeatureTransformer, self).__init__()

        self.block1 = FeatureTransformerBlock(feature_dim)
        self.block2 = FeatureTransformerBlock(feature_dim)

    def call(self, x, training=None):
        x1 = self.block1(x, training=training)
        x2 = self.block2(x1, training=training)
        return tf.sqrt(0.5) * x1 + x2


class FeatureTransformer(tf.keras.Model):
    def __init__(self, shared: SharedFeatureTransformer, feature_dim: int):
        super(FeatureTransformer, self).__init__()
        self.shared = shared
        self.block1 = FeatureTransformerBlock(feature_dim)
        self.block2 = FeatureTransformerBlock(feature_dim)

    def call(self, x, training=None):
        x = self.shared(x)
        x1 = self.block1(x, training=training)
        x1 = tf.sqrt(0.5) * x + x1
        x2 = self.block2(x1, training=training)
        return tf.sqrt(0.5) * x1 + x2


class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureTransformerBlock(feature_dim, apply_glu=False)

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return x * prior_scales
