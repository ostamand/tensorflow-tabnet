from typing import List

import tensorflow as tf
from tensorflow_addons.activations import sparsemax

from tabnet.models.utils import glu
from tabnet.models.gbn import GhostBatchNormalization


class FeatureBlock(tf.keras.Model):
    def __init__(
        self,
        feature_dim: int,
        apply_glu: bool = True,
        bn_momentum: float = 0.9,
        bn_virtual_bs: int = 512,
        fc: tf.keras.layers.Layer = None,
        epsilon: float = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
        # batch norm are not shared
        # self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, virtual_batch_size=bn_virtual_bs, epsilon=epsilon)

        # 16384 / 512 = 32
        self.bn = GhostBatchNormalization(virtual_divider=32, momentum=bn_momentum)

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        if self.apply_gpu:
            return glu(x, self.feature_dim)
        return x


class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_bs: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            bn_momentum=bn_momentum,
            bn_virtual_bs=bn_virtual_bs,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return sparsemax(x * prior_scales)


class FeatureTransformer(tf.keras.Model):
    def __init__(
        self,
        feature_dim: int,
        fcs: List[tf.keras.layers.Layer] = [],
        n_total: int = 4,
        n_shared: int = 2,
        bn_momentum: float = 0.9,
        bn_virtual_bs: int = 512,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kargs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
            "bn_virtual_bs": bn_virtual_bs,
        }

        # build blocks
        self.blocks: List[FeatureBlock] = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kargs))

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        x = self.blocks[0](x, training=training)
        for n in range(1, self.n_total):
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]
