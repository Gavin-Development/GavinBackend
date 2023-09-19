import os
import unittest
import json
import shutil

import torch.cuda

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
from GavinCore.models import TransformerIntegration, RotaryTransformerIntegration, PerformerIntegration, \
    FNetIntegration, PerformerReluIntegration, \
    PreTrainedEmbeddingTransformerIntegration, tfds, np
from GavinCore.utils import keras
from GavinCore.datasets import DatasetAPICreator
from GavinCore.callbacks import *
from GavinCore.load_data import load_tokenized_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_set_path = os.getenv('TEST_DATA_PATH')
should_use_python = bool(os.getenv('USE_PYTHON_LOADER', True))
clean_models = os.getenv("CLEAN_MODELS", False)


def get_embedding_idx(embedding_path):
    embedding_idx = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embedding_idx[word] = coefs
    return embedding_idx


def get_embedding_matrix(embedding_idx, tokenizer: tfds.deprecated.text.SubwordTextEncoder):
    embedding_matrix = np.zeros((len(tokenizer.subwords) + 1, 128))
    for i, word in enumerate(tokenizer.subwords):
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


class TestModelCallbacks(unittest.TestCase):
    model_name = {RotaryTransformerIntegration: "RotaryTransformerIntegration",
                  PreTrainedEmbeddingTransformerIntegration: "PreTrainedEmbeddingTransformerIntegration",
                  PerformerReluIntegration: "TestPerformerRelu",
                  PerformerIntegration: "TestPerformer",
                  TransformerIntegration: "TestTransformer",
                  FNetIntegration: "TestFNet"}

    glove_tokenizer = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'GloVe'))

    coef_matrix = get_embedding_matrix(
        get_embedding_idx(os.path.join(BASE_DIR, os.path.join('tests/test_files', 'vectors-128.txt'))),
        tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(BASE_DIR,
                                                                            os.path.join('tests/test_files', 'GloVe'))))

    @classmethod
    def tearDownClass(cls) -> None:
        if clean_models:
            for model_name in cls.model_name.values():
                shutil.rmtree(f"../models/{model_name}/")
        else:
            print("Models not cleaned.")

    def setUp(self) -> None:
        self.tokenizer_path = os.path.join(BASE_DIR, os.path.join('tests/test_files', 'Tokenizer-3'))
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        self.max_samples = 5_000
        self.buffer_size = 20_000
        self.batch_size = 32
        self.hparams = {TransformerIntegration: {
            'NUM_LAYERS': 1,
            'UNITS': 256,
            'D_MODEL': 128,
            'NUM_HEADS': 2,
            'DROPOUT': 0.1,
            'MAX_LENGTH': 52,
            'TOKENIZER': self.tokenizer,
            'MODEL_NAME': self.model_name[TransformerIntegration],
            'FLOAT16': False,
            'EPOCHS': 0,
            'SAVE_FREQ': 'epoch',
            'BATCH_SIZE': self.batch_size
        },
            RotaryTransformerIntegration: {
                'NUM_LAYERS': 1,
                'UNITS': 256,
                'D_MODEL': 128,
                'NUM_HEADS': 2,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 52,
                'TOKENIZER': self.tokenizer,
                'MODEL_NAME': self.model_name[RotaryTransformerIntegration],
                'FLOAT16': False,
                'EPOCHS': 0,
                'SAVE_FREQ': 'epoch',
                'BATCH_SIZE': self.batch_size
            },
            PerformerIntegration: {
                'NUM_LAYERS': 1,
                'UNITS': 256,
                'D_MODEL': 128,
                'NUM_HEADS': 2,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 52,
                'NUM_FEATURES': 128,
                'TOKENIZER': self.tokenizer,
                'MODEL_NAME': self.model_name[PerformerIntegration],
                'FLOAT16': False,
                'EPOCHS': 0,
                'SAVE_FREQ': 'epoch',
                'BATCH_SIZE': self.batch_size
            },
            FNetIntegration: {
                'NUM_LAYERS': 1,
                'UNITS': 256,
                'D_MODEL': 128,
                'NUM_HEADS': 2,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 52,
                'TOKENIZER': self.tokenizer,
                'MODEL_NAME': self.model_name[FNetIntegration],
                'FLOAT16': False,
                'EPOCHS': 0,
                'SAVE_FREQ': 'epoch',
                'BATCH_SIZE': self.batch_size
            },
            PerformerReluIntegration: {
                'NUM_LAYERS': 1,
                'UNITS': 256,
                'D_MODEL': 128,
                'NUM_HEADS': 2,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 52,
                'NUM_FEATURES': 128,
                'TOKENIZER': self.tokenizer,
                'MODEL_NAME': self.model_name[PerformerReluIntegration],
                'FLOAT16': False,
                'EPOCHS': 0,
                'SAVE_FREQ': 'epoch',
                'BATCH_SIZE': self.batch_size
            },
            PreTrainedEmbeddingTransformerIntegration: {
                'NUM_LAYERS': 1,
                'UNITS': 256,
                'D_MODEL': 128,
                'NUM_HEADS': 2,
                'DROPOUT': 0.1,
                'MAX_LENGTH': 52,
                'TOKENIZER': tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.glove_tokenizer),
                'MODEL_NAME': self.model_name[PreTrainedEmbeddingTransformerIntegration],
                'FLOAT16': False,
                'EPOCHS': 0,
                'SAVE_FREQ': 'epoch',
                'BATCH_SIZE': self.batch_size,
                'EMBEDDING_MATRIX': self.coef_matrix
            }}
        self.save_freq = 50
        self.config_for_models = {}
        for model_type, config in self.hparams.items():
            self.config_for_models[model_type] = config.copy()
            self.config_for_models[model_type] = {k.lower(): v for k, v in self.config_for_models[model_type].items()}
            self.config_for_models[model_type]['max_len'] = self.config_for_models[model_type]['max_length']
            self.config_for_models[model_type]['name'] = self.config_for_models[model_type]['model_name']
            self.config_for_models[model_type]['mixed'] = self.config_for_models[model_type]['float16']
            self.config_for_models[model_type]['base_log_dir'] = '../models/'
            del self.config_for_models[model_type]['max_length'], self.config_for_models[model_type]['model_name'], \
                self.config_for_models[model_type]['float16']
        keras.backend.clear_session()  # Reduces the amount of memory this will use.
        self.should_use_python_legacy = should_use_python
        self.should_use_cpp_legacy = False
        self.data_set_path = data_set_path

    def run_model(self, model_type, callbacks, model):
        questions, answers = load_tokenized_data(max_samples=self.max_samples,
                                                 data_path=self.data_set_path,
                                                 filename="Tokenizer-3",
                                                 s_token=model.start_token,
                                                 e_token=model.end_token, max_len=model.max_len,
                                                 cpp_legacy=self.should_use_cpp_legacy,
                                                 python_legacy=self.should_use_python_legacy)

        if self.should_use_python_legacy:
            questions = keras.preprocessing.sequence.pad_sequences(questions, maxlen=model.max_len,
                                                                   padding='post')
            answers = keras.preprocessing.sequence.pad_sequences(answers, maxlen=model.max_len,
                                                                 padding='post')

        dataset_train, _ = DatasetAPICreator.create_data_objects(questions, answers,
                                                                 buffer_size=self.buffer_size,
                                                                 batch_size=self.batch_size,
                                                                 vocab_size=model.vocab_size)
        try:
            model.fit(training_dataset=dataset_train,
                      epochs=1, callbacks=callbacks)
        except Exception as e:
            self.fail(f"Model fit failed: {e}")
        model.save_hparams()
        self.assertTrue(os.path.exists(f'../models/{self.model_name[model_type]}/config/config.json'))
        self.assertTrue(
            os.path.exists(f'../models/{self.model_name[model_type]}'
                           f'/tokenizer/{self.model_name[model_type]}_tokenizer.subwords'))
        hparams = self.hparams[model_type]
        if "EMBEDDING_MATRIX" in self.hparams[model_type]:
            del self.hparams[model_type]["EMBEDDING_MATRIX"]
        hparams['TOKENIZER'] = os.path.join(f'../models/{self.model_name[model_type]}',
                                            os.path.join('tokenizer',
                                                         f'{self.model_name[model_type]}' + '_tokenizer'))
        hparams['EPOCHS'] = hparams['EPOCHS'] + 1
        f = open(f'../models/{self.model_name[model_type]}/config/config.json')
        open_json = json.load(f)
        self.assertEqual(open_json, hparams)
        f.close()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def test_001_predict_callback(self):
        for model_type in self.model_name.keys():
            with self.subTest(msg=f"Testing {model_type.__name__}"):
                if model_type == PreTrainedEmbeddingTransformerIntegration:
                    self.skipTest("Documentation needs to be updated in Keras Core.")
                model = model_type(**self.config_for_models[model_type])
                callbacks = [PredictCallback(model.tokenizer, model.start_token, model.end_token, model.max_len,
                                             model.log_dir, model, update_freq=100)]  # Update every batches
                self.run_model(model_type, callbacks, model)

    def test_002_attention_image_logging_callback(self):
        for model_type in self.model_name.keys():
            with self.subTest(msg=f"Testing {model_type.__name__}"):
                if model_type == PreTrainedEmbeddingTransformerIntegration:
                    self.skipTest("Documentation needs to be updated in Keras Core.")
                model = model_type(**self.config_for_models[model_type])
                callbacks = [AttentionImageLoggingCallback(
                    log_dir=model.log_dir,
                    verbose=1,
                    wrapper_model=model
                )]  # Update every batches
                self.run_model(model_type, callbacks, model)

    def test_003_pythonic_archive_kit_saver(self):
        for model_type in self.model_name.keys():
            with self.subTest(msg=f"Testing {model_type.__name__}"):
                if model_type == PreTrainedEmbeddingTransformerIntegration:
                    self.skipTest("Documentation needs to be updated in Keras Core.")
                model = model_type(**self.config_for_models[model_type])
                callbacks = [PythonArchiveKitSaver(
                    save_path=os.path.join(model.log_dir, 'saved_model.pak'),
                    save_freq='epoch',
                    wrapper_model=model
                )]  # Update every batches
                self.run_model(model_type, callbacks, model)
                if not os.path.exists(os.path.join(model.log_dir, 'saved_model.pak')):
                    self.fail("Saved PAK file not found!")
