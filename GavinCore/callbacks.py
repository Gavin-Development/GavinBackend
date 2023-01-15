import random
import os
from typing import List, AnyStr, Dict, Union

import numpy as np

from .models import tf, tfds


class PredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer: tfds.deprecated.text.SubwordTextEncoder, start_token: List[int], end_token: List[int],
                 max_length: int, log_dir: AnyStr, wrapper_model, update_freq: Union[int, str] = 'epoch',
                 minimum_samples: int = 3, maximum_samples: int = 6, prompts: List[AnyStr] = None):
        super(PredictCallback, self).__init__()
        self.wrapper_model = wrapper_model
        self.tokenizer = tokenizer
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.MAX_LENGTH = max_length
        self.prompts = ["Hey?", "Hi?", "Hello.", "How are you?", "How are you doing?", "My name is Josh.",
                        "Nice to meet you!", "What is your name?", "How old are you?", "Are you married?"] \
            if prompts is None else prompts
        random.shuffle(self.prompts)
        self.log_dir = log_dir

        self.title_formatting = "=" * 10
        self.past_tests = []
        self.past_logs = []
        self.update_freq = update_freq
        self.batches = 0
        self.minimum_samples = minimum_samples
        self.maximum_samples = maximum_samples
        self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train/'))

    def _predict(self):
        predictions = []
        print("Predicting... (This could take a little bit.)")
        for i in range(random.randint(self.minimum_samples, self.maximum_samples)):
            sentence = self.prompts[i]
            prediction = self.wrapper_model.predict(sentence)

            predictions.append(
                (sentence, prediction))
        random.shuffle(self.prompts)

        return predictions

    def output_information(self, value, logs, is_epoch=True):
        tests = self._predict()
        print(f"{self.title_formatting} Responses for {'Epoch' if is_epoch else 'Step'}: "
              f"{value} | {self.wrapper_model.name} {self.title_formatting}")
        for (sentence, response) in tests:
            sentence = sentence.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            print(f"Input: {sentence}\nOutput: {response}")
            self.past_tests.append((sentence, response, f"{'Epoch' if is_epoch else 'Step'}: {value}"))
            self.past_logs.append((f"{'Epoch' if is_epoch else 'Step'}: {value}", logs))
        print(self.title_formatting + self.title_formatting + self.title_formatting)
        with self.file_writer.as_default():
            for (sentence, response) in tests:
                epoch = f"{'Epoch' if is_epoch else 'Step'}: {value}"
                with tf.name_scope("Predict Callback"):
                    text = f"""Response: {response}"""
                    tf.summary.text(f"Input: {sentence} | {epoch}", text, step=self.batches)

    def on_batch_end(self, batch, logs=None):
        self.batches += 1
        if isinstance(self.update_freq, int) and self.update_freq != 0:
            if self.batches % self.update_freq == 0:
                self.output_information(is_epoch=False, value=self.batches, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if isinstance(self.update_freq, str):
            if self.update_freq == "epoch":
                self.output_information(value=epoch, logs=logs)


class AttentionImageLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir: AnyStr, verbose: int = 0, update_freq: Union[int, str] = 'epoch',
                 wrapper_model=None):
        super(AttentionImageLoggingCallback, self).__init__()
        if wrapper_model is None:
            raise ValueError("Wrapper model must be passed to AttentionImageLoggingCallback.")
        self.model = wrapper_model.model
        self.verbose = verbose
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.batches = 0
        self.attention_layers = self._get_attention_layers()
        self.file_writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train/'))
        self.wrapper_model = wrapper_model

    def _get_attention_layers(self):
        encoder_layers_names = [layer.name for layer in self.model.get_layer('encoder').layers if 'encoder_layer_' in layer.name]
        attention_layers_names = []
        for encoder_layers_name in encoder_layers_names:
            attention_layers_names.extend([(encoder_layers_name, layer.name) for layer in self.model.get_layer('encoder').get_layer(encoder_layers_name).layers if 'multi_head_attention' in layer.name])
        if self.verbose > 0:
            tf.print(f"Found {len(attention_layers_names)} attention layers.")
            tf.print(f"Attention layers: {attention_layers_names}")
        return attention_layers_names

    def _log_attention_images(self, value, logs):
        for encoder_decoder_name, attention_name in self.attention_layers:
            if 'encoder' in encoder_decoder_name:
                image_matrix = self.model.get_layer('encoder').get_layer(encoder_decoder_name).get_layer(attention_name).saved_attention_image
            else:
                image_matrix = self.model.get_layer('decoder').get_layer(encoder_decoder_name).get_layer(attention_name).saved_attention_image
            image_matrix = tf.transpose(image_matrix, perm=[0, 3, 2, 1])
            if self.verbose > 0:
                tf.print(f"Saving attention image for {encoder_decoder_name} | {attention_name}", end=" ")
                tf.print(f"Image shape: {image_matrix.shape}")
            with self.file_writer.as_default():
                with tf.name_scope("Attention Image"):
                    tf.summary.image(f"{encoder_decoder_name} | {attention_name}", image_matrix, step=self.batches)

    def on_batch_begin(self, batch, logs=None):
        self.batches += 1
        if isinstance(self.update_freq, int) and self.update_freq != 0:
            if self.batches % self.update_freq == 0:
                self._log_attention_images(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if isinstance(self.update_freq, str):
            if self.update_freq == "epoch":
                self._log_attention_images(value=epoch, logs=logs)


# Source: https://www.tensorflow.org/guide/keras/custom_callback#usage_of_selfmodel_attribute
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

      Arguments:
          patience: Number of epochs to wait after min has been hit. After this
          number of no improvement, training stops.
      """

    def __init__(self, patience: int = 0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        # Init self vars
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None

    def on_train_begin(self, logs: Dict = None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                self.model.save_weights(self)

    def on_train_end(self, logs: Dict = None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
