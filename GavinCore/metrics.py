from .utils import keras, torch


class Precision(keras.metrics.Precision):

    def __init__(self, max_len: int, from_logits: bool = False, **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.max_len = max_len
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = torch.reshape(y_true, shape=(-1, self.max_len))
        super(Precision, self).update_state(
            y_true, y_pred if not self.from_logits else torch.argmax(y_pred, dim=2),
            sample_weight=sample_weight)

    def result(self):
        super(Precision, self).result()


# @tf.keras.utils.register_keras_serializable('GavinCore')
class Perplexity(keras.metrics.Metric):
    def __init__(self, max_len: int, vocab_size: int, **kwargs):
        super(Perplexity, self).__init__(**kwargs)
        self.max_len = max_len
        self.perplexity = self.add_weight(name='p', initializer="zeros")
        self.vocab_size = vocab_size
        self.scce = keras.losses.SparseCategoricalCrossentropy(
            reduction='none', from_logits=True)

    def result(self):
        return self.perplexity

    def update_state(self, y_true, y_pred, sample_weight=None, numerical_stabiliser=5e-10):
        """
        Args:
            :param y_true: (batch_size, max_len)
            :param y_pred: (batch_size, max_len, vocab_size)
            :param sample_weight: torch.Tensor
            :param numerical_stabiliser:
                Stabiliser to prevent log(0) errors.
        :return:
        """
        y_true = keras.ops.cast(y_true, y_pred.dtype)

        loss = self.scce(y_true, y_pred) + numerical_stabiliser
        loss = torch.exp(loss.mean())
        self.perplexity.assign(loss)

    def get_config(self):
        config = {"max_len": self.max_len, "vocab_size": self.vocab_size}
        return config
