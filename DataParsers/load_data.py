import base64
import pickle
import typing
import sys
import os
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm

root_path = Path(__file__).resolve().parent.parent
WINDOWS = False
if "windows" in platform.system().lower():
    sys.path.append(os.path.join(str(root_path), 'CustomPackages/windows'))
    WINDOWS = True


# noinspection PickleLoad
def tokenized_read_thread(path: typing.AnyStr, reddit_set_max: int, s_token: typing.List[int],
                          e_token: typing.List[int], thread_id: int = 0):
    lines = []
    pbar = tqdm.tqdm(total=reddit_set_max // 2, desc=f"Thread: {thread_id}")
    with open(path, "r") as f:
        for i in range(reddit_set_max // 2):
            line = next(f).strip("'b'")
            line = line.strip("'\n'")
            line = line.strip("'")
            # line = preprocess_sentence(line)
            line = pickle.loads(base64.b64decode(line))
            line.insert(0, s_token[0])
            line.append(e_token[0])
            lines.append(line)
            pbar.update(1)
    return lines


def load_tokenized_data(max_samples: int, data_path: typing.AnyStr, filename: typing.AnyStr,
                        s_token: typing.List[int], e_token: typing.List[int], max_len: int = None,
                        python_legacy: bool = False,
                        cpp_legacy=False, single_thread=True) -> \
        typing.Tuple[typing.List[str], typing.List[str]] or typing.Tuple[np.ndarray, np.ndarray]:
    """Load tokenized data from the data files:
    {data_path}{filename}.from
    {data_path}{filename}.to these will be configurable eventually."""
    if not python_legacy and max_len is None:
        raise Exception("Max Length can't be none when Legacy is false.")
    if not WINDOWS and not python_legacy:
        raise Exception(
            "This package is only compiled for windows, linux compatability "
            "coming soon. Please use python_legacy for now.")
    if python_legacy:
        if not single_thread:
            with ProcessPoolExecutor(2) as executor:
                inputs_fn = executor.submit(tokenized_read_thread, f"{data_path}{filename}.from", max_samples,
                                            s_token, e_token, 0)
                outputs_fn = executor.submit(tokenized_read_thread, f"{data_path}{filename}.to", max_samples, s_token,
                                             e_token, 1)
                executor.shutdown()

            return inputs_fn.result(), outputs_fn.result()
        else:
            inputs = tokenized_read_thread(f"{data_path}{filename}.from", max_samples,
                                           s_token, e_token, 0)
            outputs = tokenized_read_thread(f"{data_path}{filename}.to", max_samples,
                                            s_token, e_token, 1)
            return inputs, outputs
    else:
        import GavinBackendDatasetUtils
        files = os.listdir(data_path)
        if f"{filename}-from.BIN" in files and f"{filename}-to.BIN" in files and not cpp_legacy:
            if not single_thread:
                inputs = GavinBackendDatasetUtils.LoadTrainDataMT(max_samples // 2, data_path, f"{filename}-from.BIN",
                                                                  s_token[0], e_token[0], max_len, 0)
                outputs = GavinBackendDatasetUtils.LoadTrainDataMT(max_samples // 2, data_path, f"{filename}-to.BIN",
                                                                   s_token[0], e_token[0], max_len, 0)
            else:
                inputs = GavinBackendDatasetUtils.LoadTrainDataST(max_samples // 2, data_path, f"{filename}-from.BIN",
                                                                  s_token[0], e_token[0], max_len, 0)
                outputs = GavinBackendDatasetUtils.LoadTrainDataST(max_samples // 2, data_path, f"{filename}-to.BIN",
                                                                   s_token[0], e_token[0], max_len, 0)
        elif f"{filename}.from" in files and f"{filename}.to" in files and cpp_legacy:
            inputs = GavinBackendDatasetUtils.LoadTrainDataST_Legacy(max_samples // 2, f"{data_path}",
                                                                     f"{filename}.from", s_token[0], e_token[0],
                                                                     max_len, 0)
            outputs = GavinBackendDatasetUtils.LoadTrainDataST_Legacy(max_samples // 2, f"{data_path}",
                                                                      f"{filename}.to", s_token[0], e_token[0],
                                                                      max_len, 0)
            inputs = np.asarray(inputs)
            outputs = np.asarray(outputs)
        else:
            raise FileNotFoundError(
                f"Couldn't find appropriate files for {filename} did you mean to load in python_legacy mode?")
        return inputs, outputs
