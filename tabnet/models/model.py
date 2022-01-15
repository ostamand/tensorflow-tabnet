from typing import List, Tuple

import tensorflow as tf

from tabnet.models.transformers import (
    FeatureTransformer,
    AttentiveTransformer,
)


class TabNet(tf.keras.Model):
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        output_dim: int,
        feature_columns: List = None,
        n_step: int = 1,
        n_total: int = 4,
        n_shared: int = 2,
        relaxation_factor: float = 1.5,
        bn_epsilon: float = 1e-5,
        bn_momentum: float = 0.7,
        bn_virtual_divider: int = 1,
    ):
        """TabNet

        Will output a vector of size output_dim.

        Args:
            num_features (int): Number of features.
            feature_dim (int): Embedding feature dimention to use.
            output_dim (int): Output dimension.
            feature_columns (List, optional): If defined will add a DenseFeatures layer first. Defaults to None.
            n_step (int, optional): Total number of steps. Defaults to 1.
            n_total (int, optional): Total number of feature transformer blocks. Defaults to 4.
            n_shared (int, optional): Number of shared feature transformer blocks. Defaults to 2.
            relaxation_factor (float, optional): >1 will allow features to be used more than once. Defaults to 1.5.
            bn_epsilon (float, optional): Batch normalization, epsilon. Defaults to 1e-5.
            bn_momentum (float, optional): Batch normalization, momentum. Defaults to 0.7.
            bn_virtual_divider (int, optional): Batch normalization. Full batch will be divided by this.
        """
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.feature_columns = feature_columns

        if feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns)

        # ? Switch to Ghost Batch Normalization
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms: List[FeatureTransformer] = [
            FeatureTransformer(**kargs)
        ]
        self.attentive_transforms: List[AttentiveTransformer] = []
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features, bn_momentum, bn_virtual_divider)
            )

    def call(
        self, features: tf.Tensor, training: bool = None, alpha: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.feature_columns is not None:
            features = self.input_features(features)

        bs = tf.shape(features)[0]
        out_agg = tf.zeros((bs, self.output_dim))
        prior_scales = tf.ones((bs, self.num_features))
        masks = []

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](
                masked_features, training=training, alpha=alpha
            )

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                x_for_mask = x[:, self.output_dim:]

                mask_values = self.attentive_transforms[step_i](
                    [x_for_mask, prior_scales], training=training, alpha=alpha
                )

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= self.relaxation_factor - mask_values

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )

                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

        loss = total_entropy / self.n_step

        return out_agg, loss, masks
