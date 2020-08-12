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
        n_step: int = 1,
        relaxation_factor: float = 1.5,
        epsilon: float = 1e-5,
        sparsity_coefficient: float = 1e-5,
        bn_momentum: float = 0.7,
        bn_virtual_bs: int = 512,
    ):
        """TabNet: Attentive Interpretable Tabular Learning

        Args:
            feature_dim (int): N_a
            output_dim (int): N_d
            n_step (int, optional): N_steps. Defaults to 1.
            relaxation_factor (float, optional): gamma. Defaults to 1.5.
        """
        super(TabNet, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.epsilon, self.sparsity_coefficient = epsilon, sparsity_coefficient

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, virtual_batch_size=bn_virtual_bs
        )

        # build feature transformer blocks
        shared = SharedFeatureTransformer(
            feature_dim + output_dim, bn_momentum, bn_virtual_bs,
        )

        self.feature_transforms: List[FeatureTransformer] = []
        self.attentive_transforms: List[AttentiveTransformer] = []
        for _ in range(n_step + 1):
            self.feature_transforms.append(
                FeatureTransformer(
                    shared, feature_dim + output_dim, bn_momentum, bn_virtual_bs
                )
            )
        for _ in range(n_step):
            self.attentive_transforms.append(
                AttentiveTransformer(num_features, bn_momentum, bn_virtual_bs)
            )

    def call(self, features, training=None):
        bs = tf.shape(features)[0]
        out_aggregated = tf.zeros((bs, self.output_dim))
        prior_scales = tf.ones((bs, self.num_features))

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            x = self.feature_transforms[step_i](masked_features, training=training)

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_aggregated += out

            x_for_mask = x[:, self.output_dim :]

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )
                mask_values = sparsemax(mask_values, axis=-1)

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= self.relaxation_factor - mask_values

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1
                    )
                ) / (
                    tf.cast(self.n_step, tf.float32)
                )  #! was self.n_step -1 in baseline inplementation

        if training:
            self.add_loss(self.sparsity_coefficient * total_entropy)

        return out_aggregated
