from typing import List
import tensorflow as tf

from tabnet.models.utils import sparsemax
from tabnet.models.transformers import (
    FeatureTransformer,
    SharedFeatureTransformer,
    AttentiveTransformer,
)


class TabNet(tf.keras.Model):
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        output_dim: int,
        feature_columns = None,
        n_step: int = 1,
        n_total: int = 4,
        n_shared: int = 2,
        relaxation_factor: float = 1.5,
        sparsity_coefficient: float = 1e-5,
        epsilon: float = 1e-5,
        bn_momentum: float = 0.7,
        bn_virtual_bs: int = 512,
    ):
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.epsilon = epsilon
        self.feature_columns = feature_columns
        self.sparsity_coefficient = sparsity_coefficient

        if feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns)

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=1e-5 # virtual_batch_size=bn_virtual_bs,
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum,
            "bn_virtual_bs": bn_virtual_bs,
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
                AttentiveTransformer(num_features, bn_momentum, bn_virtual_bs)
            )

    def call(self, features: tf.Tensor, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.input_features is not None:
            features = self.input_features(features)

        bs = tf.shape(features)[0]
        out_agg = tf.zeros((bs, self.output_dim))
        prior_scales = tf.ones((bs, self.num_features))

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](masked_features, training=training)

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                x_for_mask = x[:, self.output_dim :]

                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= (self.relaxation_factor - mask_values)

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1
                    )
                )

        if training:
            self.add_loss(-self.sparsity_coefficient * total_entropy / self.n_step)

        return out_agg
