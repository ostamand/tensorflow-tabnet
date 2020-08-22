import tensorflow as tf
import tensorflow.keras.backend as K


class TensorBoardWithLR(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": self.model.optimizer.lr(epoch)})
        return super().on_epoch_end(epoch, logs=logs)

