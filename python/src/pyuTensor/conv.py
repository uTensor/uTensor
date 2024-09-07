import numpy as np

from . import _pyuTensor as _C
from .util import export as _export


@_export
def conv2d(input: np.ndarray, kernel: np.ndarray,  strides: list, bias = None, padding: str = "VALID") -> np.ndarray:
    if bias is None:
        bias = np.zeros(kernel.shape[0])
    return _C.conv2d_f(input, kernel, bias, strides[:4], padding)

@_export
def max_pool2d(input: np.ndarray, kernel_size: list, strides: list, padding: str = "VALID") -> np.ndarray:
    if input.dtype == np.float32:
        return _C.max_pool_f(input, kernel_size, strides, padding)
    elif input.dtype == np.int8:
        return _C.max_pool_i8(input, kernel_size, strides, padding)
    elif input.dtype == np.uint8:
        return _C.max_pool_u8(input, kernel_size, strides, padding)
    else:
        raise ValueError(f"Unsupported data type: {input.dtype}")