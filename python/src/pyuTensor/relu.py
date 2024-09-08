import numpy as np

from . import _pyuTensor as _C
from .util import export as _export


@_export
def relu(arr: np.ndarray) -> np.ndarray:
    assert arr.dtype in [np.float32, np.float64], "Expecting array of type float32 or float64"
    return _C.relu_f(arr.astype(np.float32))

@_export
def relu_q(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Quantized ReLU
    """
    if arr.dtype == np.int8:
        return _C.relu_i8(arr, scale, zero_point)
    elif arr.dtype == np.int16:
        return _C.relu_i16(arr, scale, zero_point)
    elif arr.dtype == np.int32:
        return _C.relu_i32(arr, scale, zero_point)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")