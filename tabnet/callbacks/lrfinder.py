from typing import Text
from math import log10

import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(
        self,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        num_steps: int = 100,
        monitor: Text = "loss",
        figname: Text = "lrfinder.png",
    ):
        super(LRFinder, self).__init__()
        self.monitor = monitor
        self.num_steps = num_steps
        self.min_lr, self.max_lr = min_lr, max_lr
        self.figname = figname

        # log(y) = m * log(x) + b
        # m = log(y2/y1) / log(x2/x1)
        # b = log(y2) - m * log(x2)
        self.m = log10(max_lr / min_lr) / log10(num_steps - 1)
        self.b = log10(max_lr) - self.m * log10(num_steps - 1)
        self.__reset()

    def __reset(self):
        self.losses = []
        self.lrs = []

    def set_lr(self, step: int) -> float:
        lr = pow(10, self.m * log10(step) + self.b)
        K.set_value(self.model.optimizer.lr, lr)
        return lr

    def save_fig(self):
        plt.semilogx(self.lrs, self.losses)
        plt.title("LR Finder")
        plt.xlabel("lr")
        plt.ylabel("batch loss")
        plt.savefig(self.figname)

    def on_train_batch_end(self, batch, logs=None):
        it = len(self.losses) + 1

        if len(self.losses) < self.num_steps:
            self.losses.append(logs[self.monitor])

        if len(self.lrs) < self.num_steps:
            self.lrs.append(self.set_lr(it))
        else:
            self.model.stop_training = True
            if self.figname is not None:
                self.save_fig()

    def on_train_begin(self, logs=None):
        self.__reset()
        self.lrs.append(self.set_lr(1))
