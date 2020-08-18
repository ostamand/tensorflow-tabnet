import tensorflow as tf


# taken from https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])