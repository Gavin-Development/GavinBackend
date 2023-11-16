import unittest
import sys
import os
import numpy as np
from GavinCore.load_data import load_tokenized_data
from GavinCore.datasets import DatasetAPICreator, DatasetDirectFromFileAPICreator
from GavinCore.models import keras
from GavinCore.utils import tf

data_set_path = os.getenv('TEST_DATA_PATH')
url_set_path = os.getenv('TEST_URL_PATH')


class DataLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.max_samples = 10_000
        self.start_token = [69908]
        self.end_token = [69909]
        self.max_len = 52
        self.DATA_PATH = data_set_path
        self.URL_PATH = url_set_path
        self.batch_size = 64
        self.buffer_size = 20_000
        self.path_to = os.path.join(self.DATA_PATH, "Tokenizer-3-{}.BIN")

    def test_001_legacy_load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.DATA_PATH,
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, python_legacy=True)
        except Exception as e:
            self.fail(f"Legacy failed: {e}")
        self.assertEqual(len(questions), self.max_samples // 2)
        self.assertEqual(len(answers), self.max_samples // 2)
        self.assertEqual(type(answers), list)
        self.assertEqual(type(questions), list)

    def test_002_dll_download(self):
        # remove import to test
        # this will trigger the download of the .dlls
        from GavinCore.load_data import load_tokenized_data
        load_tokenized_data(max_samples=self.max_samples,
                            data_path=self.DATA_PATH,
                            filename="Tokenizer-3",
                            s_token=self.start_token,
                            e_token=self.end_token, max_len=50, single_thread=True)

    def test_003_CustomPackage_load_single_thread(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.DATA_PATH,
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=True)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(questions))
        self.assertGreaterEqual(self.max_samples // 2, len(questions))

        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(answers))
        self.assertGreaterEqual(self.max_samples // 2, len(answers))

        self.assertEqual(np.ndarray, type(questions),
                         msg=f"type of questions is not of type {np.ndarray} but of type {type(questions)}")
        self.assertEqual(np.ndarray, type(answers),
                         msg=f"type of answers is not of type {np.ndarray} but of type {type(answers)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(questions),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(questions)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(questions),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(questions)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(answers),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(answers)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(answers),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(answers)}")

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_004_CustomPackage_load_multi_thread(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.DATA_PATH,
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=False)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(questions))
        self.assertGreaterEqual(self.max_samples // 2, len(questions))

        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(answers))
        self.assertGreaterEqual(self.max_samples // 2, len(answers))

        self.assertEqual(np.ndarray, type(questions),
                         msg=f"type of questions is not of type {np.ndarray} but of type {type(questions)}")
        self.assertEqual(np.ndarray, type(answers),
                         msg=f"type of answers is not of type {np.ndarray} but of type {type(answers)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(questions),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(questions)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(questions),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(questions)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(answers),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(answers)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(answers),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(answers)}")

    def test_003_CustomPackage_Legacy_Load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.DATA_PATH,
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=False)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(questions))
        self.assertGreaterEqual(self.max_samples // 2, len(questions))

        self.assertLessEqual(((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), len(answers))
        self.assertGreaterEqual(self.max_samples // 2, len(answers))

        self.assertEqual(np.ndarray, type(questions),
                         msg=f"type of questions is not of type {np.ndarray} but of type {type(questions)}")
        self.assertEqual(np.ndarray, type(answers),
                         msg=f"type of answers is not of type {np.ndarray} but of type {type(answers)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(questions),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(questions)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(questions),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(questions)}")

        self.assertLessEqual((((self.max_samples // 2) - ((self.max_samples // 2) * 0.05)), self.max_len),
                             np.shape(answers),
                             msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                 f"but of size {np.size(answers)}")
        self.assertGreaterEqual((self.max_samples // 2, self.max_len), np.shape(answers),
                                msg=f"Questions is not of size {(self.max_samples // 2, self.max_len)} "
                                    f"but of size {np.size(answers)}")

    def test_004_URLDownload_Legacy_Load(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.URL_PATH,
                                                     filename="Tokenizer-3-Discord",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=False,
                                                     python_legacy=True)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertEqual(len(questions), self.max_samples // 2)
        self.assertEqual(len(answers), self.max_samples // 2)
        self.assertEqual(type(answers), list)
        self.assertEqual(type(questions), list)

    def test_005_DataLoadCreator(self):
        try:
            questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                     data_path=self.URL_PATH,
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=False,
                                                     python_legacy=True)
            questions = keras.preprocessing.sequence.pad_sequences(questions, maxlen=self.max_len,
                                                                   padding='post')
            answers = keras.preprocessing.sequence.pad_sequences(answers, maxlen=self.max_len,
                                                                 padding='post')
            dataset_train, dataset_val = DatasetAPICreator.create_data_objects(questions, answers,
                                                                               buffer_size=self.buffer_size,
                                                                               batch_size=self.batch_size,
                                                                               vocab_size=66901)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
            return
        for first_batch in dataset_train:
            inputs = first_batch[0]['inputs']
            dec_inputs = first_batch[0]['dec_inputs']
            outputs = first_batch[1]['outputs']
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(inputs)))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(dec_inputs)))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(outputs)))

            inputs_dtype = inputs.numpy().dtype
            dec_inputs_dtype = dec_inputs.numpy().dtype
            outputs_dtype = outputs.numpy().dtype
            self.assertEqual(np.int32, inputs_dtype)
            self.assertEqual(np.int32, dec_inputs_dtype)
            self.assertEqual(np.int32, outputs_dtype)
            break
        for first_batch in dataset_val:
            inputs = first_batch[0]['inputs']
            dec_inputs = first_batch[0]['dec_inputs']
            outputs = first_batch[1]['outputs']
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(inputs)))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(dec_inputs)))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(outputs)))

            inputs_dtype = inputs.numpy().dtype
            dec_inputs_dtype = dec_inputs.numpy().dtype
            outputs_dtype = outputs.numpy().dtype
            self.assertEqual(np.int32, inputs_dtype)
            self.assertEqual(np.int32, dec_inputs_dtype)
            self.assertEqual(np.int32, outputs_dtype)
            break

    # noinspection StrFormat
    def test_006_DatasetFromFileString(self):
        try:
            dataset_train, dataset_val = DatasetDirectFromFileAPICreator.create_data_objects(
                questions_file=self.path_to.format("from"),
                answers_file=self.path_to.format("to"),
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                vocab_size=66901,
                max_length=self.max_len,
                number_of_samples=self.max_samples // 2,
                start_token=self.start_token[0],
                end_token=self.end_token[0],
                padding_value=0)

        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
            return
        for first_batch in dataset_train:
            inputs = first_batch[0]['inputs']
            dec_inputs = first_batch[0]['dec_inputs']
            outputs = first_batch[1]['outputs']
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(inputs).numpy()))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(dec_inputs).numpy()))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(outputs).numpy()))

            inputs_dtype = inputs.numpy().dtype
            dec_inputs_dtype = dec_inputs.numpy().dtype
            outputs_dtype = outputs.numpy().dtype
            self.assertEqual(np.int32, inputs_dtype)
            self.assertEqual(np.int32, dec_inputs_dtype)
            self.assertEqual(np.int32, outputs_dtype)
            break
        for first_batch in dataset_val:
            inputs = first_batch[0]['inputs']
            dec_inputs = first_batch[0]['dec_inputs']
            outputs = first_batch[1]['outputs']
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(inputs).numpy()))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(dec_inputs).numpy()))
            self.assertEqual((self.batch_size, self.max_len), tuple(tf.shape(outputs).numpy()))

            inputs_dtype = inputs.numpy().dtype
            dec_inputs_dtype = dec_inputs.numpy().dtype
            outputs_dtype = outputs.numpy().dtype
            self.assertEqual(np.int32, inputs_dtype)
            self.assertEqual(np.int32, dec_inputs_dtype)
            self.assertEqual(np.int32, outputs_dtype)
            break
