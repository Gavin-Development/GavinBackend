from .models import tfds, tf
from .load_data import *  # Ensures GavinBackendDatasetUtils can load
if IS_SUPPORTED_VERSION:
    import GavinBackendDatasetUtils as LTD
else:
    import GavinBackend.GavinCore.empty_classes as LTD


class DatasetAPICreator:
    def __init__(self, questions: list, answers: list, buffer_size: int, batch_size: int, vocab_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.questions_train = questions
        self.answers_train = answers
        self.vocab_size = vocab_size

    @classmethod
    def create_data_objects(cls, questions: list, answers: list, buffer_size: int, batch_size: int, vocab_size: int):
        self = cls(questions, answers, buffer_size, batch_size, vocab_size)

        dec_inputs_train = self.answers_train.copy()
        dec_inputs_train[:, -1] = 0
        outputs_train = self.answers_train.copy()
        outputs_train[:, 0] = 0
        outputs_train = np.roll(outputs_train.copy(), -1)  # Roll back values -1 to not leave an empty value.
        del self.answers_train
        # decoder inputs use the previous target as input
        # remove s_token from targets
        # print("Beginning Dataset Shuffling, Batching and Prefetch.")
        dataset_all = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': self.questions_train,  # Source
                'dec_inputs': dec_inputs_train  # Targets
            },
            {
                'outputs': outputs_train  # Outputs
            }))
        dataset_t = dataset_all.take(int(len(self.questions_train) * .8))
        dataset_v = dataset_all.skip(int(len(self.questions_train) * .8))
        del dataset_all

        dataset_t = dataset_t.shuffle(self.buffer_size)
        dataset_v = dataset_v.shuffle(self.buffer_size)
        dataset_t = dataset_t.batch(self.batch_size)
        dataset_v = dataset_v.batch(self.batch_size)

        dataset_v = dataset_v.cache()
        dataset_t = dataset_t.cache()
        dataset_v = dataset_v.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_t = dataset_t.prefetch(tf.data.experimental.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset_t = dataset_t.with_options(options)
        dataset_v = dataset_v.with_options(options)

        return dataset_t, dataset_v


class DatasetDirectFromFileAPICreator:
    def __init__(self, questions_file: typing.Union[LTD.BINFile, str], answers_file: typing.Union[LTD.BINFile, str],
                 buffer_size: int, batch_size: int, vocab_size: int, max_length: int, number_of_samples: int,
                 start_token: int = None, end_token: int = None, padding_value: int = None):
        if (isinstance(questions_file, str) and
            isinstance(answers_file, str)) and \
                (start_token is None or
                 end_token is None or
                 padding_value is None):
            raise ValueError("If you are using strings for the files, you must provide max_length, start_token, "
                             "end_token and padding_value.")
        if isinstance(questions_file, str):
            self.questions_bin_file = LTD.BINFile(questions_file, start_token, end_token, max_length, padding_value)
        if isinstance(answers_file, str):
            self.answers_bin_file = LTD.BINFile(answers_file, start_token, end_token, max_length, padding_value)
        else:
            self.questions_bin_file = questions_file
            self.answers_bin_file = answers_file

        self.legacy = True if hasattr(self.questions_bin_file, 'MaxNumberOfSamples') else False
        questions_max = self.questions_bin_file.MaxNumberOfSamples if self.legacy else \
            self.questions_bin_file.max_number_of_samples
        answers_max = self.answers_bin_file.MaxNumberOfSamples if self.legacy else \
            self.answers_bin_file.max_number_of_samples

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        if number_of_samples > questions_max or number_of_samples > answers_max:
            raise ValueError(f"Number of samples is greater than the number of samples"
                             f"({min(answers_max, questions_max)})"
                             f" in the file.")
        self.number_of_samples = number_of_samples

    def numpy_generator(self):
        current_index = 0
        while current_index + 1 < self.number_of_samples:
            questions = self.questions_bin_file[current_index]
            answers = self.answers_bin_file[current_index]

            dec_inputs = answers.copy()
            outputs = answers.copy()
            del answers

            dec_inputs[-1] = 0

            outputs[0] = 0
            outputs = np.roll(outputs.copy(), -1)  # Roll back values -1 to not leave an empty value.

            return_data = ({'inputs': questions, 'dec_inputs': dec_inputs}, {'outputs': outputs})
            current_index += 1
            yield return_data

    @classmethod
    def create_data_objects(cls, questions_file: typing.Union[LTD.BINFile, str],
                            answers_file: typing.Union[LTD.BINFile, str],
                            buffer_size: int, batch_size: int, vocab_size: int,
                            max_length: int, number_of_samples: int,
                            start_token: int = None, end_token: int = None, padding_value: int = None):
        self = cls(questions_file, answers_file, buffer_size, batch_size, vocab_size, max_length, number_of_samples,
                   start_token, end_token, padding_value)

        dataset_all = tf.data.Dataset.from_generator(self.numpy_generator,
                                                     output_types=({'inputs': tf.int32,
                                                                    'dec_inputs': tf.int32},
                                                                   {'outputs': tf.int32}),
                                                     output_shapes=({'inputs': (self.max_length,),
                                                                     'dec_inputs': (self.max_length,)},
                                                                    {'outputs': (self.max_length,)}))
        dataset_t = dataset_all.take(int(self.number_of_samples * .8))
        dataset_v = dataset_all.skip(int(self.number_of_samples * .8))
        del dataset_all

        dataset_t = dataset_t.batch(self.batch_size)
        dataset_v = dataset_v.batch(self.batch_size)
        dataset_t = dataset_t.shuffle(self.buffer_size)
        dataset_v = dataset_v.shuffle(self.buffer_size)

        dataset_v = dataset_v.cache()
        dataset_t = dataset_t.cache()
        dataset_v = dataset_v.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_t = dataset_t.prefetch(tf.data.experimental.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset_t = dataset_t.with_options(options)
        dataset_v = dataset_v.with_options(options)

        return dataset_t, dataset_v
