from typing import List
import tensorflow as tf

from tabnet.utils import glu, sparsemax


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

        self.bn = tf.keras.layers.BatchNormalization()

        # build feature transformer blocks
        shared = SharedFeatureTransformer(feature_dim + output_dim)
        self.feature_transforms: List[FeatureTransformer] = []
        self.attentive_transforms: List[AttentiveTransformer] = []
        for _ in range(n_step + 1):
            self.feature_transforms.append(
                FeatureTransformer(shared, feature_dim + output_dim)
            )
        for _ in range(n_step):
            self.attentive_transforms.append(AttentiveTransformer(num_features))

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
