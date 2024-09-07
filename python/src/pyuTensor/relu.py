import numpy as np

from . import _pyuTensor as _C
from .util import export as _export


@_export
def relu(arr: np.ndarray):
    if arr.dtype in [np.float32, np.float64]:
        return _C.relu_f(arr.astype(np.float32))
    elif arr.dtype == np.int8:
        return _C.relu_i8(arr)
    elif arr.dtype == np.int16:
        return _C.relu_i16(arr)
    elif arr.dtype == np.int32:
        return _C.relu_i32(arr)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")