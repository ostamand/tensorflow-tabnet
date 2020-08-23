import os
import random
import tensorflow as tf
import numpy as np


def set_seed(seed: int = 42):
    # reference: https://github.com/NVIDIA/framework-determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
