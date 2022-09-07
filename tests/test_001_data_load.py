import unittest
import sys
import os
import numpy as np
from DataParsers.load_data import load_tokenized_data
from pathlib import Path

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

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
    def test_002_dll_download(self):
        # remove import to test
        # this will trigger the download of the .dlls
        from DataParsers.load_data import load_tokenized_data
        load_tokenized_data(max_samples=self.max_samples,
                            data_path=self.DATA_PATH,
                            filename="Tokenizer-3",
                            s_token=self.start_token,
                            e_token=self.end_token, max_len=50, single_thread=True)

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
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

    @unittest.skipUnless(sys.platform.startswith("win"), "requires Windows")
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
                                                     filename="Tokenizer-3",
                                                     s_token=self.start_token,
                                                     e_token=self.end_token, max_len=self.max_len, single_thread=False, python_legacy=True)
        except Exception as e:
            self.fail(f"Custom Load failed: {e}")
        self.assertEqual(len(questions), self.max_samples // 2)
        self.assertEqual(len(answers), self.max_samples // 2)
        self.assertEqual(type(answers), list)
        self.assertEqual(type(questions), list)

