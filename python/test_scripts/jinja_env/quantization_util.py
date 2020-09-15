import numpy as np

def quantize(x, zp, scale, symmetric, narrow_range=False, num_quant_bits=8):
  if x.dtype != np.float32:
    raise ValueError("Quantize expects float32 input")
  if symmetric:
    if num_quant_bits == 8:
        ttype = np.int8
    else:
        ttype = np.int32
  else:
    if num_quant_bits == 8:
        ttype = np.uint8
    else:
        ttype = np.uint32
  limits = np.iinfo(ttype)
  lmax = limits.max
  lmin = limits.min if not narrow_range else limits.min + 1
  # Use clipping to handle the overflow cases wrapping around a ring
  quantized = np.clip((x/scale + zp), a_min=lmin, a_max=lmax).astype(ttype)
  return quantized


def dequantize(x, zp, scale):
  if x.dtype != np.int8 and x.dtype != np.uint8 and x.dtype != np.int32 and x.dtype != np.uint32:
    raise ValueError("Quantize expects integer input")
  limits = np.iinfo(x.dtype)
  return (x - zp)*scale

def get_quantization_params(x, symmetric=True, per_channel_quantization=False, narrow_range=False, num_quant_bits=8):
  if per_channel_quantization:
    raise NotImplementedError
  if symmetric:
    if num_quant_bits == 8:
        ttype = np.int8
    else:
        ttype = np.int32
  else:
    if num_quant_bits == 8:
        ttype = np.uint8
    else:
        ttype = np.uint32
  limits = np.iinfo(ttype)
  lmax = limits.max
  # If narrow range and symmetric -> [-127, 127]
  lmin = limits.min if not narrow_range else limits.min + 1
  xmin = np.min(x)
  xmax = np.max(x)

  zp = 0
  if symmetric:
    # Find scales that dont excede ranges and dont distort 0
    min_scale = lmin / xmin if (lmin * xmin) > 0 else np.finfo(np.float32).max
    max_scale = lmax / xmax if (lmax * xmax) > 0 else np.finfo(np.float32).max
    if min_scale < max_scale:
      scale = min_scale
      iscale = xmin / lmin
      xmax = lmax * iscale
    else:
      scale = max_scale
      iscale = xmax / lmax
      xmin = lmin * iscale

    # Flip the scales to match quantization affine map convention
    # R = (Q - ZP)*scale < -- > Q = R/scale + zp
    scale = 1.0/scale
    iscale = 1.0/iscale

  else:
    scale = (xmax - xmin) / (lmax - lmin)
    # Use the limits min of int8/uint8 to compute the initial real value of the zero point
    initialZp  = lmin - xmin/scale
    if initialZp < lmin:
      zp = lmin
    elif initialZp > lmax:
      zp = lmax
    else:
      zp = int(round(initialZp))

  #quantized = quantize(x, zp, scale, symmetric, num_quant_bits)
  #return (zp, scale, quantized)
  return (zp, scale)


