from ._pyuTensor import *
from ._version import __version__
import numpy as np

def relu(arr: np.ndarray):
    if arr.dtype in [np.float32, np.float64]:
        return relu_f(arr.astype(np.float32))
    raise ValueError(f"Unsupported dtype: {arr.dtype}")
    