import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class DecayWithWarmupSchedule(LearningRateSchedule):

    def __init__(self, learning_rate, min_learning_rate, warmup, decay_rate, decay_steps):
        super(DecayWithWarmupSchedule, self).__init__()
        self.learning_rate, self.min_learning_rate = learning_rate, min_learning_rate
        self.warmup = warmup
        self.decay_rate, self.decay_steps = decay_rate, decay_steps

        self.m = (learning_rate - min_learning_rate) / warmup
        self.b = learning_rate - self.m * warmup

    def __call__(self, step):
        return tf.cond(
            tf.greater_equal(step, self.warmup),
            lambda: self.learning_rate * tf.pow(self.decay_rate, (step / self.decay_steps)),
            lambda: self.m * step + self.b
        )

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup": self.warmup,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps,
        }
