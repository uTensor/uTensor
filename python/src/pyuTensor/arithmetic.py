import numpy as np

from . import _pyuTensor as _C
from .util import export as _export


@_export
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if b.dtype != np.float32:
        b = b.astype(np.float32)
    return _C.mul_kernel(a, b)

@_export
def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype != b.dtype:
        raise ValueError(f"Data types must match: {a.dtype} v.s {b.dtype}")
    if a.dtype == np.float32:
        return _C.add_kernel_f(a, b)
    elif a.dtype == np.int8:
        return _C.add_kernel_i8(a, b)
    elif a.dtype == np.uint8:
        return _C.add_kernel_u8(a, b)
    elif a.dtype == np.int16:
        return _C.add_kernel_i16(a, b)
    elif a.dtype == np.uint16:
        return _C.add_kernel_u16(a, b)
    elif a.dtype == np.int32:
        return _C.add_kernel_i32(a, b)
    elif a.dtype == np.uint32:
        return _C.add_kernel_u32(a, b)
    else:
        raise ValueError(f"Unsupported data type: {a.dtype}")