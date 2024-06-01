from ._pyuTensor import *
from ._version import __version__
import numpy as np

def relu(arr: np.ndarray):
    if arr.dtype == np.float32:
        return relu_f(arr)
    elif arr.dtype == np.int32:
        return relu_i32(arr)
    raise ValueError(f"Unsupported dtype: {arr.dtype}")
    