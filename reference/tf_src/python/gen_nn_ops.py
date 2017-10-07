"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections as _collections

from google.protobuf import text_format as _text_format

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2

# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library

def _avg_pool(value, ksize, strides, padding, data_format=None, name=None):
  r"""Performs average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `value`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of `value`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    The average pooled output tensor.
  """
  result = _op_def_lib.apply_op("AvgPool", value=value, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def avg_pool3d(input, ksize, strides, padding, data_format=None, name=None):
  r"""Performs 3D average pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The average pooled output tensor.
  """
  result = _op_def_lib.apply_op("AvgPool3D", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _avg_pool3d_grad(orig_input_shape, grad, ksize, strides, padding,
                     data_format=None, name=None):
  r"""Computes gradients of average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      The original input dimensions.
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`. The backprop for input.
  """
  result = _op_def_lib.apply_op("AvgPool3DGrad",
                                orig_input_shape=orig_input_shape, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _avg_pool_grad(orig_input_shape, grad, ksize, strides, padding,
                   data_format=None, name=None):
  r"""Computes gradients of the average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      1-D.  Shape of the original input to `avg_pool`.
    grad: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
      the output of `avg_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of the input.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
    4-D.  Gradients w.r.t. the input of `avg_pool`.
  """
  result = _op_def_lib.apply_op("AvgPoolGrad",
                                orig_input_shape=orig_input_shape, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _batch_norm_with_global_normalization(t, m, v, beta, gamma,
                                          variance_epsilon,
                                          scale_after_normalization,
                                          name=None):
  r"""Batch normalization.

  This op is deprecated. Prefer `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  result = _op_def_lib.apply_op("BatchNormWithGlobalNormalization", t=t, m=m,
                                v=v, beta=beta, gamma=gamma,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return result



__batch_norm_with_global_normalization_grad_outputs = ["dx", "dm", "dv", "db",
                                                      "dg"]
_BatchNormWithGlobalNormalizationGradOutput = _collections.namedtuple(
    "BatchNormWithGlobalNormalizationGrad",
    __batch_norm_with_global_normalization_grad_outputs)


def _batch_norm_with_global_normalization_grad(t, m, v, gamma, backprop,
                                               variance_epsilon,
                                               scale_after_normalization,
                                               name=None):
  r"""Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).

    dx: A `Tensor`. Has the same type as `t`. 4D backprop tensor for input.
    dm: A `Tensor`. Has the same type as `t`. 1D backprop tensor for mean.
    dv: A `Tensor`. Has the same type as `t`. 1D backprop tensor for variance.
    db: A `Tensor`. Has the same type as `t`. 1D backprop tensor for beta.
    dg: A `Tensor`. Has the same type as `t`. 1D backprop tensor for gamma.
  """
  result = _op_def_lib.apply_op("BatchNormWithGlobalNormalizationGrad", t=t,
                                m=m, v=v, gamma=gamma, backprop=backprop,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return _BatchNormWithGlobalNormalizationGradOutput._make(result)



def _bias_add(value, bias, data_format=None, name=None):
  r"""Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    Broadcasted sum of `value` and `bias`.
  """
  result = _op_def_lib.apply_op("BiasAdd", value=value, bias=bias,
                                data_format=data_format, name=name)
  return result



def bias_add_grad(out_backprop, data_format=None, name=None):
  r"""The backward operation for "BiasAdd" on the "bias" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
    1-D with size the feature dimension of `out_backprop`.
  """
  result = _op_def_lib.apply_op("BiasAddGrad", out_backprop=out_backprop,
                                data_format=data_format, name=name)
  return result



def _bias_add_v1(value, bias, name=None):
  r"""Adds `bias` to `value`.

  This is a deprecated version of BiasAdd and will be soon removed.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    Broadcasted sum of `value` and `bias`.
  """
  result = _op_def_lib.apply_op("BiasAddV1", value=value, bias=bias,
                                name=name)
  return result



def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
           data_format=None, name=None):
  r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: A list of `ints`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
        `data_format`, see below for details.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A 4-D tensor. The dimension order is determined by the value of
    `data_format`, see below for details.
  """
  result = _op_def_lib.apply_op("Conv2D", input=input, filter=filter,
                                strides=strides, padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result



def conv2d_backprop_filter(input, filter_sizes, out_backprop, strides,
                           padding, use_cudnn_on_gpu=None, data_format=None,
                           name=None):
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. 4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
    the `filter` input of the convolution.
  """
  result = _op_def_lib.apply_op("Conv2DBackpropFilter", input=input,
                                filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result



def conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding,
                          use_cudnn_on_gpu=None, data_format=None, name=None):
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `float32`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
    4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
    w.r.t. the input of the convolution.
  """
  result = _op_def_lib.apply_op("Conv2DBackpropInput",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result



def conv3d(input, filter, strides, padding, data_format=None, name=None):
  r"""Computes a 3-D convolution given 5-D `input` and `filter` tensors.

  In signal processing, cross-correlation is a measure of similarity of
  two waveforms as a function of a time-lag applied to one of them. This
  is also known as a sliding dot product or sliding inner-product.

  Our Conv3D implements a form of cross-correlation.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, in_depth, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_depth, filter_height, filter_width, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3D", input=input, filter=filter,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def conv3d_backprop_filter(input, filter, out_backprop, strides, padding,
                           name=None):
  r"""Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropFilter", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, padding=padding, name=name)
  return result



def conv3d_backprop_filter_v2(input, filter_sizes, out_backprop, strides,
                              padding, data_format=None, name=None):
  r"""Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 5-D
      `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
      tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropFilterV2", input=input,
                                filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, data_format=data_format,
                                name=name)
  return result



def conv3d_backprop_input(input, filter, out_backprop, strides, padding,
                          name=None):
  r"""Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropInput", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, padding=padding, name=name)
  return result



def conv3d_backprop_input_v2(input_sizes, filter, out_backprop, strides,
                             padding, data_format=None, name=None):
  r"""Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropInputV2",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, data_format=data_format,
                                name=name)
  return result



def depthwise_conv2d_native(input, filter, strides, padding, data_format=None,
                            name=None):
  r"""Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNative", input=input,
                                filter=filter, strides=strides,
                                padding=padding, data_format=data_format,
                                name=name)
  return result



def depthwise_conv2d_native_backprop_filter(input, filter_sizes, out_backprop,
                                            strides, padding,
                                            data_format=None, name=None):
  r"""Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape based on `data_format`.  For example, if
      `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
      in_width, in_channels]` tensor.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. 4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
    the `filter` input of the convolution.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNativeBackpropFilter",
                                input=input, filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, data_format=data_format,
                                name=name)
  return result



def depthwise_conv2d_native_backprop_input(input_sizes, filter, out_backprop,
                                           strides, padding, data_format=None,
                                           name=None):
  r"""Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`, based
      on `data_format`.  For example, if `data_format` is 'NHWC' then
       `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, channels, height, width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
    4-D with shape according to `data_format`.  For example, if
    `data_format` is 'NHWC', output shape is `[batch, in_height,
    in_width, in_channels]`.  Gradient w.r.t. the input of the
    convolution.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNativeBackpropInput",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, data_format=data_format,
                                name=name)
  return result



def dilation2d(input, filter, strides, rates, padding, name=None):
  r"""Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filter` is equal to the
  negation of the erosion of `-input` by the reflected `filter`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    4-D with shape `[batch, out_height, out_width, depth]`.
  """
  result = _op_def_lib.apply_op("Dilation2D", input=input, filter=filter,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result



def dilation2d_backprop_filter(input, filter, out_backprop, strides, rates,
                               padding, name=None):
  r"""Computes the gradient of morphological 2-D dilation with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    3-D with shape `[filter_height, filter_width, depth]`.
  """
  result = _op_def_lib.apply_op("Dilation2DBackpropFilter", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result



def dilation2d_backprop_input(input, filter, out_backprop, strides, rates,
                              padding, name=None):
  r"""Computes the gradient of morphological 2-D dilation with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    4-D with shape `[batch, in_height, in_width, depth]`.
  """
  result = _op_def_lib.apply_op("Dilation2DBackpropInput", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result



def elu(features, name=None):
  r"""Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Elu", features=features, name=name)
  return result



def _elu_grad(gradients, outputs, name=None):
  r"""Computes gradients for the exponential linear (Elu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Elu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Elu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients * (outputs + 1)` if outputs < 0,
    `gradients` otherwise.
  """
  result = _op_def_lib.apply_op("EluGrad", gradients=gradients,
                                outputs=outputs, name=name)
  return result



_fractional_avg_pool_outputs = ["output", "row_pooling_sequence",
                               "col_pooling_sequence"]
_FractionalAvgPoolOutput = _collections.namedtuple(
    "FractionalAvgPool", _fractional_avg_pool_outputs)


def fractional_avg_pool(value, pooling_ratio, pseudo_random=None,
                        overlapping=None, deterministic=None, seed=None,
                        seed2=None, name=None):
  r"""Performs fractional average pooling on the input.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalAvgPool node in the computation graph. Mainly used
      in unit test to make FractionalAvgPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`. output tensor after fractional avg pooling.
    row_pooling_sequence: A `Tensor` of type `int64`. row pooling sequence, needed to calculate gradient.
    col_pooling_sequence: A `Tensor` of type `int64`. column pooling sequence, needed to calculate gradient.
  """
  result = _op_def_lib.apply_op("FractionalAvgPool", value=value,
                                pooling_ratio=pooling_ratio,
                                pseudo_random=pseudo_random,
                                overlapping=overlapping,
                                deterministic=deterministic, seed=seed,
                                seed2=seed2, name=name)
  return _FractionalAvgPoolOutput._make(result)



def _fractional_avg_pool_grad(orig_input_tensor_shape, out_backprop,
                              row_pooling_sequence, col_pooling_sequence,
                              overlapping=None, name=None):
  r"""Computes gradient of the FractionalAvgPool function.

  Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
  FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
  out_backprop to those indices that form the same pooling cell. Therefore, we
  just need to know the shape of original input tensor, instead of the whole
  tensor.

  Args:
    orig_input_tensor_shape: A `Tensor` of type `int64`.
      Original input tensor shape for `fractional_avg_pool`
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_avg_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
    4-D.  Gradients w.r.t. the input of `fractional_avg_pool`.
  """
  result = _op_def_lib.apply_op("FractionalAvgPoolGrad",
                                orig_input_tensor_shape=orig_input_tensor_shape,
                                out_backprop=out_backprop,
                                row_pooling_sequence=row_pooling_sequence,
                                col_pooling_sequence=col_pooling_sequence,
                                overlapping=overlapping, name=name)
  return result



_fractional_max_pool_outputs = ["output", "row_pooling_sequence",
                               "col_pooling_sequence"]
_FractionalMaxPoolOutput = _collections.namedtuple(
    "FractionalMaxPool", _fractional_max_pool_outputs)


def fractional_max_pool(value, pooling_ratio, pseudo_random=None,
                        overlapping=None, deterministic=None, seed=None,
                        seed2=None, name=None):
  r"""Performs fractional max pooling on the input.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word "fractional", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly uniform.
  For example, let's look at the height dimension, and the constraints on the
  list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  For more details on fractional max pooling, see this paper:
  [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalMaxPool node in the computation graph. Mainly used
      in unit test to make FractionalMaxPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`. output tensor after fractional max pooling.
    row_pooling_sequence: A `Tensor` of type `int64`. row pooling sequence, needed to calculate gradient.
    col_pooling_sequence: A `Tensor` of type `int64`. column pooling sequence, needed to calculate gradient.
  """
  result = _op_def_lib.apply_op("FractionalMaxPool", value=value,
                                pooling_ratio=pooling_ratio,
                                pseudo_random=pseudo_random,
                                overlapping=overlapping,
                                deterministic=deterministic, seed=seed,
                                seed2=seed2, name=name)
  return _FractionalMaxPoolOutput._make(result)



def _fractional_max_pool_grad(orig_input, orig_output, out_backprop,
                              row_pooling_sequence, col_pooling_sequence,
                              overlapping=None, name=None):
  r"""Computes gradient of the FractionalMaxPool function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Original input for `fractional_max_pool`
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      Original output for `fractional_max_pool`
    out_backprop: A `Tensor`. Must have the same type as `orig_input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_max_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
    4-D.  Gradients w.r.t. the input of `fractional_max_pool`.
  """
  result = _op_def_lib.apply_op("FractionalMaxPoolGrad",
                                orig_input=orig_input,
                                orig_output=orig_output,
                                out_backprop=out_backprop,
                                row_pooling_sequence=row_pooling_sequence,
                                col_pooling_sequence=col_pooling_sequence,
                                overlapping=overlapping, name=name)
  return result



__fused_batch_norm_outputs = ["y", "batch_mean", "batch_variance",
                             "reserve_space_1", "reserve_space_2"]
_FusedBatchNormOutput = _collections.namedtuple(
    "FusedBatchNorm", __fused_batch_norm_outputs)


def _fused_batch_norm(x, scale, offset, mean, variance, epsilon=None,
                      data_format=None, is_training=None, name=None):
  r"""Batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string`. Defaults to `"NHWC"`.
      The data format for x and y. Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`. A 4D Tensor for output data.
    batch_mean: A `Tensor`. Has the same type as `x`. A 1D Tensor for the computed batch mean, to be used by TensorFlow
      to compute the running mean.
    batch_variance: A `Tensor`. Has the same type as `x`. A 1D Tensor for the computed batch variance, to be used by
      TensorFlow to compute the running variance.
    reserve_space_1: A `Tensor`. Has the same type as `x`. A 1D Tensor for the computed batch mean, to be reused
      in the gradient computation.
    reserve_space_2: A `Tensor`. Has the same type as `x`. A 1D Tensor for the computed batch variance (inverted variance
      in the cuDNN case), to be used in the gradient computation.
  """
  result = _op_def_lib.apply_op("FusedBatchNorm", x=x, scale=scale,
                                offset=offset, mean=mean, variance=variance,
                                epsilon=epsilon, data_format=data_format,
                                is_training=is_training, name=name)
  return _FusedBatchNormOutput._make(result)



_fused_batch_norm_grad_outputs = ["x_backprop", "scale_backprop",
                                 "offset_backprop", "reserve_space_3",
                                 "reserve_space_4"]
_FusedBatchNormGradOutput = _collections.namedtuple(
    "FusedBatchNormGrad", _fused_batch_norm_grad_outputs)


def fused_batch_norm_grad(y_backprop, x, scale, reserve_space_1,
                          reserve_space_2, epsilon=None, data_format=None,
                          is_training=None, name=None):
  r"""Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for the computed batch mean, to be reused
      in the gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for the computed batch variance (inverted variance
      in the cuDNN case), to be used in the gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`. A 4D Tensor for the gradient with respect to x.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`. A 1D Tensor for the gradient with respect to scale.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`. A 1D Tensor for the gradient with respect to offset.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`. Unused placeholder to match the mean input in FusedBatchNorm.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`. Unused placeholder to match the variance input
      in FusedBatchNorm.
  """
  result = _op_def_lib.apply_op("FusedBatchNormGrad", y_backprop=y_backprop,
                                x=x, scale=scale,
                                reserve_space_1=reserve_space_1,
                                reserve_space_2=reserve_space_2,
                                epsilon=epsilon, data_format=data_format,
                                is_training=is_training, name=name)
  return _FusedBatchNormGradOutput._make(result)



def fused_pad_conv2d(input, paddings, filter, mode, strides, padding,
                     name=None):
  r"""Performs a padding as a preprocess during a convolution.

  Similar to FusedResizeAndPadConv2d, this op allows for an optimized
  implementation where the spatial padding transformation stage is fused with the
  im2col lookup, but in this case without the bilinear filtering required for
  resizing. Fusing the padding prevents the need to write out the intermediate
  results as whole tensors, reducing memory pressure, and we can get some latency
  gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
  order is used instead.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("FusedPadConv2D", input=input,
                                paddings=paddings, filter=filter, mode=mode,
                                strides=strides, padding=padding, name=name)
  return result



def fused_resize_and_pad_conv2d(input, size, paddings, filter, mode, strides,
                                padding, resize_align_corners=None,
                                name=None):
  r"""Performs a resize and padding as a preprocess during a convolution.

  It's often possible to do spatial transformations more efficiently as part of
  the packing stage of a convolution, so this op allows for an optimized
  implementation where these stages are fused together. This prevents the need to
  write out the intermediate results as whole tensors, reducing memory pressure,
  and we can get some latency gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and defaults to
  'NHWC' order.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    resize_align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale input by (new_height - 1) / (height - 1),
      which exactly aligns the 4 corners of images and resized images. If false, rescale
      by new_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("FusedResizeAndPadConv2D", input=input,
                                size=size, paddings=paddings, filter=filter,
                                mode=mode, strides=strides, padding=padding,
                                resize_align_corners=resize_align_corners,
                                name=name)
  return result



def in_top_k(predictions, targets, k, name=None):
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.
  """
  result = _op_def_lib.apply_op("InTopK", predictions=predictions,
                                targets=targets, k=k, name=name)
  return result



def l2_loss(t, name=None):
  r"""L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`. 0-D.
  """
  result = _op_def_lib.apply_op("L2Loss", t=t, name=name)
  return result



def lrn(input, depth_radius=None, bias=None, alpha=None, beta=None,
        name=None):
  r"""Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("LRN", input=input, depth_radius=depth_radius,
                                bias=bias, alpha=alpha, beta=beta, name=name)
  return result



def _lrn_grad(input_grads, input_image, output_image, depth_radius=None,
              bias=None, alpha=None, beta=None, name=None):
  r"""Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_grads`. The gradients for LRN.
  """
  result = _op_def_lib.apply_op("LRNGrad", input_grads=input_grads,
                                input_image=input_image,
                                output_image=output_image,
                                depth_radius=depth_radius, bias=bias,
                                alpha=alpha, beta=beta, name=name)
  return result



def _log_softmax(logits, name=None):
  r"""Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  """
  result = _op_def_lib.apply_op("LogSoftmax", logits=logits, name=name)
  return result



def _max_pool(input, ksize, strides, padding, data_format=None, name=None):
  r"""Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The max pooled output tensor.
  """
  result = _op_def_lib.apply_op("MaxPool", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def max_pool3d(input, ksize, strides, padding, data_format=None, name=None):
  r"""Performs 3D max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The max pooled output tensor.
  """
  result = _op_def_lib.apply_op("MaxPool3D", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _max_pool3d_grad(orig_input, orig_output, grad, ksize, strides, padding,
                     data_format=None, name=None):
  r"""Computes gradients of max pooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `float32`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  result = _op_def_lib.apply_op("MaxPool3DGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _max_pool3d_grad_grad(orig_input, orig_output, grad, ksize, strides,
                          padding, data_format=None, name=None):
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
    Gradients of gradients w.r.t. the input to `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPool3DGradGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _max_pool_grad(orig_input, orig_output, grad, ksize, strides, padding,
                   data_format=None, name=None):
  r"""Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
    Gradients w.r.t. the input to `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _max_pool_grad_grad(orig_input, orig_output, grad, ksize, strides,
                        padding, data_format=None, name=None):
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
    Gradients of gradients w.r.t. the input to `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGradGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result



def _max_pool_grad_grad_with_argmax(input, grad, argmax, ksize, strides,
                                    padding, name=None):
  r"""Computes second-order gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      input of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Gradients of gradients w.r.t. the input of `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGradGradWithArgmax", input=input,
                                grad=grad, argmax=argmax, ksize=ksize,
                                strides=strides, padding=padding, name=name)
  return result



def _max_pool_grad_with_argmax(input, grad, argmax, ksize, strides, padding,
                               name=None):
  r"""Computes gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      output of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Gradients w.r.t. the input of `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGradWithArgmax", input=input,
                                grad=grad, argmax=argmax, ksize=ksize,
                                strides=strides, padding=padding, name=name)
  return result



_max_pool_with_argmax_outputs = ["output", "argmax"]
_MaxPoolWithArgmaxOutput = _collections.namedtuple(
    "MaxPoolWithArgmax", _max_pool_with_argmax_outputs)


def max_pool_with_argmax(input, ksize, strides, padding, Targmax=None,
                         name=None):
  r"""Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index
  `((b * height + y) * width + x) * channels + c`.

  The indices returned are always in `[0, height) x [0, width)` before flattening,
  even if padding is involved and the mathematically correct answer is outside
  (either negative or too large).  This is a bug, but fixing it is difficult to do
  in a safe backwards compatible way, especially due to flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`. The max pooled output tensor.
    argmax: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.
  """
  result = _op_def_lib.apply_op("MaxPoolWithArgmax", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                Targmax=Targmax, name=name)
  return _MaxPoolWithArgmaxOutput._make(result)



_quantized_avg_pool_outputs = ["output", "min_output", "max_output"]
_QuantizedAvgPoolOutput = _collections.namedtuple(
    "QuantizedAvgPool", _quantized_avg_pool_outputs)


def quantized_avg_pool(input, min_input, max_input, ksize, strides, padding,
                       name=None):
  r"""Produces the average pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      4-D with shape `[batch, height, width, channels]`.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.  The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedAvgPool", input=input,
                                min_input=min_input, max_input=max_input,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return _QuantizedAvgPoolOutput._make(result)



_quantized_batch_norm_with_global_normalization_outputs = ["result",
                                                          "result_min",
                                                          "result_max"]
_QuantizedBatchNormWithGlobalNormalizationOutput = _collections.namedtuple(
    "QuantizedBatchNormWithGlobalNormalization",
    _quantized_batch_norm_with_global_normalization_outputs)


def quantized_batch_norm_with_global_normalization(t, t_min, t_max, m, m_min,
                                                   m_max, v, v_min, v_max,
                                                   beta, beta_min, beta_max,
                                                   gamma, gamma_min,
                                                   gamma_max, out_type,
                                                   variance_epsilon,
                                                   scale_after_normalization,
                                                   name=None):
  r"""Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).

    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("QuantizedBatchNormWithGlobalNormalization",
                                t=t, t_min=t_min, t_max=t_max, m=m,
                                m_min=m_min, m_max=m_max, v=v, v_min=v_min,
                                v_max=v_max, beta=beta, beta_min=beta_min,
                                beta_max=beta_max, gamma=gamma,
                                gamma_min=gamma_min, gamma_max=gamma_max,
                                out_type=out_type,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return _QuantizedBatchNormWithGlobalNormalizationOutput._make(result)



_quantized_bias_add_outputs = ["output", "min_out", "max_out"]
_QuantizedBiasAddOutput = _collections.namedtuple(
    "QuantizedBiasAdd", _quantized_bias_add_outputs)


def quantized_bias_add(input, bias, min_input, max_input, min_bias, max_bias,
                       out_type, name=None):
  r"""Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).

    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_out: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedBiasAdd", input=input, bias=bias,
                                min_input=min_input, max_input=max_input,
                                min_bias=min_bias, max_bias=max_bias,
                                out_type=out_type, name=name)
  return _QuantizedBiasAddOutput._make(result)



_quantized_conv2d_outputs = ["output", "min_output", "max_output"]
_QuantizedConv2DOutput = _collections.namedtuple(
    "QuantizedConv2D", _quantized_conv2d_outputs)


def quantized_conv2d(input, filter, min_input, max_input, min_filter,
                     max_filter, strides, padding, out_type=None, name=None):
  r"""Computes a 2D convolution given quantized 4D input and filter tensors.

  The inputs are quantized tensors where the lowest value represents the real
  number of the associated minimum, and the highest represents the maximum.
  This means that you can only interpret the quantized output in the same way, by
  taking the returned minimum and maximum values into account.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      filter's input_depth dimension must match input's depth dimensions.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the lowest quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the highest quantized filter value represents.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedConv2D", input=input, filter=filter,
                                min_input=min_input, max_input=max_input,
                                min_filter=min_filter, max_filter=max_filter,
                                strides=strides, padding=padding,
                                out_type=out_type, name=name)
  return _QuantizedConv2DOutput._make(result)



_quantized_max_pool_outputs = ["output", "min_output", "max_output"]
_QuantizedMaxPoolOutput = _collections.namedtuple(
    "QuantizedMaxPool", _quantized_max_pool_outputs)


def quantized_max_pool(input, min_input, max_input, ksize, strides, padding,
                       name=None):
  r"""Produces the max pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor. The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedMaxPool", input=input,
                                min_input=min_input, max_input=max_input,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return _QuantizedMaxPoolOutput._make(result)



_quantized_relu_outputs = ["activations", "min_activations",
                          "max_activations"]
_QuantizedReluOutput = _collections.namedtuple(
    "QuantizedRelu", _quantized_relu_outputs)


def quantized_relu(features, min_features, max_features, out_type=None,
                   name=None):
  r"""Computes Quantized Rectified Linear: `max(features, 0)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedRelu", features=features,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedReluOutput._make(result)



_quantized_relu6_outputs = ["activations", "min_activations",
                           "max_activations"]
_QuantizedRelu6Output = _collections.namedtuple(
    "QuantizedRelu6", _quantized_relu6_outputs)


def quantized_relu6(features, min_features, max_features, out_type=None,
                    name=None):
  r"""Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedRelu6", features=features,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedRelu6Output._make(result)



_quantized_relu_x_outputs = ["activations", "min_activations",
                            "max_activations"]
_QuantizedReluXOutput = _collections.namedtuple(
    "QuantizedReluX", _quantized_relu_x_outputs)


def quantized_relu_x(features, max_value, min_features, max_features,
                     out_type=None, name=None):
  r"""Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedReluX", features=features,
                                max_value=max_value,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedReluXOutput._make(result)



def relu(features, name=None):
  r"""Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Relu", features=features, name=name)
  return result



def _relu6(features, name=None):
  r"""Computes rectified linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Relu6", features=features, name=name)
  return result



def _relu6_grad(gradients, features, name=None):
  r"""Computes rectified linear 6 gradients for a Relu6 operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding Relu6 operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu6 operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`. The gradients:
    `gradients * (features > 0) * (features < 6)`.
  """
  result = _op_def_lib.apply_op("Relu6Grad", gradients=gradients,
                                features=features, name=name)
  return result



def _relu_grad(gradients, features, name=None):
  r"""Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    `gradients * (features > 0)`.
  """
  result = _op_def_lib.apply_op("ReluGrad", gradients=gradients,
                                features=features, name=name)
  return result



def _softmax(logits, name=None):
  r"""Computes softmax activations.

  For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  """
  result = _op_def_lib.apply_op("Softmax", logits=logits, name=name)
  return result



__softmax_cross_entropy_with_logits_outputs = ["loss", "backprop"]
_SoftmaxCrossEntropyWithLogitsOutput = _collections.namedtuple(
    "SoftmaxCrossEntropyWithLogits",
    __softmax_cross_entropy_with_logits_outputs)


def _softmax_cross_entropy_with_logits(features, labels, name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must have the same type as `features`.
      batch_size x num_classes matrix
      The caller must ensure that each batch of labels represents a valid
      probability distribution.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`. Per example loss (batch_size vector).
    backprop: A `Tensor`. Has the same type as `features`. backpropagated gradients (batch_size x num_classes matrix).
  """
  result = _op_def_lib.apply_op("SoftmaxCrossEntropyWithLogits",
                                features=features, labels=labels, name=name)
  return _SoftmaxCrossEntropyWithLogitsOutput._make(result)



def softplus(features, name=None):
  r"""Computes softplus: `log(exp(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Softplus", features=features, name=name)
  return result



def _softplus_grad(gradients, features, name=None):
  r"""Computes softplus gradients for a softplus operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding softplus operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softplus operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients / (1 + exp(-features))`.
  """
  result = _op_def_lib.apply_op("SoftplusGrad", gradients=gradients,
                                features=features, name=name)
  return result



def softsign(features, name=None):
  r"""Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Softsign", features=features, name=name)
  return result



def _softsign_grad(gradients, features, name=None):
  r"""Computes softsign gradients for a softsign operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding softsign operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softsign operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients / (1 + abs(-features)) ** 2`.
  """
  result = _op_def_lib.apply_op("SoftsignGrad", gradients=gradients,
                                features=features, name=name)
  return result



__sparse_softmax_cross_entropy_with_logits_outputs = ["loss", "backprop"]
_SparseSoftmaxCrossEntropyWithLogitsOutput = _collections.namedtuple(
    "SparseSoftmaxCrossEntropyWithLogits",
    __sparse_softmax_cross_entropy_with_logits_outputs)


def _sparse_softmax_cross_entropy_with_logits(features, labels, name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`. Per example loss (batch_size vector).
    backprop: A `Tensor`. Has the same type as `features`. backpropagated gradients (batch_size x num_classes matrix).
  """
  result = _op_def_lib.apply_op("SparseSoftmaxCrossEntropyWithLogits",
                                features=features, labels=labels, name=name)
  return _SparseSoftmaxCrossEntropyWithLogitsOutput._make(result)



__top_k_outputs = ["values", "indices"]
_TopKOutput = _collections.namedtuple(
    "TopK", __top_k_outputs)


def _top_k(input, k, sorted=None, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  If `k` varies dynamically, use `TopKV2` below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      1-D or higher with last dimension at least `k`.
    k: An `int` that is `>= 0`.
      Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`. The `k` largest elements along each last dimensional slice.
    indices: A `Tensor` of type `int32`. The indices of `values` within the last dimension of `input`.
  """
  result = _op_def_lib.apply_op("TopK", input=input, k=k, sorted=sorted,
                                name=name)
  return _TopKOutput._make(result)



__top_kv2_outputs = ["values", "indices"]
_TopKV2Output = _collections.namedtuple(
    "TopKV2", __top_kv2_outputs)


def _top_kv2(input, k, sorted=None, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor` of type `int32`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`. The `k` largest elements along each last dimensional slice.
    indices: A `Tensor` of type `int32`. The indices of `values` within the last dimension of `input`.
  """
  result = _op_def_lib.apply_op("TopKV2", input=input, k=k, sorted=sorted,
                                name=name)
  return _TopKV2Output._make(result)


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AvgPool"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "AvgPool3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "AvgPool3DGrad"
  input_arg {
    name: "orig_input_shape"
    type: DT_INT32
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "AvgPoolGrad"
  input_arg {
    name: "orig_input_shape"
    type: DT_INT32
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchNormWithGlobalNormalization"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "m"
    type_attr: "T"
  }
  input_arg {
    name: "v"
    type_attr: "T"
  }
  input_arg {
    name: "beta"
    type_attr: "T"
  }
  input_arg {
    name: "gamma"
    type_attr: "T"
  }
  output_arg {
    name: "result"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "variance_epsilon"
    type: "float"
  }
  attr {
    name: "scale_after_normalization"
    type: "bool"
  }
  deprecation {
    version: 9
    explanation: "Use tf.nn.batch_normalization()"
  }
}
op {
  name: "BatchNormWithGlobalNormalizationGrad"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "m"
    type_attr: "T"
  }
  input_arg {
    name: "v"
    type_attr: "T"
  }
  input_arg {
    name: "gamma"
    type_attr: "T"
  }
  input_arg {
    name: "backprop"
    type_attr: "T"
  }
  output_arg {
    name: "dx"
    type_attr: "T"
  }
  output_arg {
    name: "dm"
    type_attr: "T"
  }
  output_arg {
    name: "dv"
    type_attr: "T"
  }
  output_arg {
    name: "db"
    type_attr: "T"
  }
  output_arg {
    name: "dg"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "variance_epsilon"
    type: "float"
  }
  attr {
    name: "scale_after_normalization"
    type: "bool"
  }
  deprecation {
    version: 9
    explanation: "Use tf.nn.batch_normalization()"
  }
}
op {
  name: "BiasAdd"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "bias"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "BiasAddGrad"
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "BiasAddV1"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "bias"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Conv2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv2DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv2DBackpropInput"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
}
op {
  name: "Conv3DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  deprecation {
    version: 10
    explanation: "Use Conv3DBackpropFilterV2"
  }
}
op {
  name: "Conv3DBackpropFilterV2"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
}
op {
  name: "Conv3DBackpropInput"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  deprecation {
    version: 10
    explanation: "Use Conv3DBackpropInputV2"
  }
}
op {
  name: "Conv3DBackpropInputV2"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
}
op {
  name: "DepthwiseConv2dNative"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "DepthwiseConv2dNativeBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "DepthwiseConv2dNativeBackpropInput"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Dilation2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "Dilation2DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "filter_backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "Dilation2DBackpropInput"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "in_backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "Elu"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "EluGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "outputs"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "FractionalAvgPool"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "row_pooling_sequence"
    type: DT_INT64
  }
  output_arg {
    name: "col_pooling_sequence"
    type: DT_INT64
  }
  attr {
    name: "pooling_ratio"
    type: "list(float)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "pseudo_random"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "overlapping"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "deterministic"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "FractionalAvgPoolGrad"
  input_arg {
    name: "orig_input_tensor_shape"
    type: DT_INT64
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "row_pooling_sequence"
    type: DT_INT64
  }
  input_arg {
    name: "col_pooling_sequence"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "overlapping"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "FractionalMaxPool"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "row_pooling_sequence"
    type: DT_INT64
  }
  output_arg {
    name: "col_pooling_sequence"
    type: DT_INT64
  }
  attr {
    name: "pooling_ratio"
    type: "list(float)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "pseudo_random"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "overlapping"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "deterministic"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "FractionalMaxPoolGrad"
  input_arg {
    name: "orig_input"
    type_attr: "T"
  }
  input_arg {
    name: "orig_output"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "row_pooling_sequence"
    type: DT_INT64
  }
  input_arg {
    name: "col_pooling_sequence"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "overlapping"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "FusedBatchNorm"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "scale"
    type_attr: "T"
  }
  input_arg {
    name: "offset"
    type_attr: "T"
  }
  input_arg {
    name: "mean"
    type_attr: "T"
  }
  input_arg {
    name: "variance"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "batch_mean"
    type_attr: "T"
  }
  output_arg {
    name: "batch_variance"
    type_attr: "T"
  }
  output_arg {
    name: "reserve_space_1"
    type_attr: "T"
  }
  output_arg {
    name: "reserve_space_2"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "epsilon"
    type: "float"
    default_value {
      f: 0.0001
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
  }
  attr {
    name: "is_training"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "FusedBatchNormGrad"
  input_arg {
    name: "y_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "scale"
    type_attr: "T"
  }
  input_arg {
    name: "reserve_space_1"
    type_attr: "T"
  }
  input_arg {
    name: "reserve_space_2"
    type_attr: "T"
  }
  output_arg {
    name: "x_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "scale_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "offset_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "reserve_space_3"
    type_attr: "T"
  }
  output_arg {
    name: "reserve_space_4"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "epsilon"
    type: "float"
    default_value {
      f: 0.0001
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
  }
  attr {
    name: "is_training"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "FusedPadConv2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "mode"
    type: "string"
    allowed_values {
      list {
        s: "REFLECT"
        s: "SYMMETRIC"
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "FusedResizeAndPadConv2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "resize_align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "mode"
    type: "string"
    allowed_values {
      list {
        s: "REFLECT"
        s: "SYMMETRIC"
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "InTopK"
  input_arg {
    name: "predictions"
    type: DT_FLOAT
  }
  input_arg {
    name: "targets"
    type_attr: "T"
  }
  output_arg {
    name: "precision"
    type: DT_BOOL
  }
  attr {
    name: "k"
    type: "int"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "L2Loss"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "LRN"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "depth_radius"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "alpha"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "beta"
    type: "float"
    default_value {
      f: 0.5
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_HALF
      }
    }
  }
}
op {
  name: "LRNGrad"
  input_arg {
    name: "input_grads"
    type_attr: "T"
  }
  input_arg {
    name: "input_image"
    type_attr: "T"
  }
  input_arg {
    name: "output_image"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "depth_radius"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "alpha"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "beta"
    type: "float"
    default_value {
      f: 0.5
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_HALF
      }
    }
  }
}
op {
  name: "LogSoftmax"
  input_arg {
    name: "logits"
    type_attr: "T"
  }
  output_arg {
    name: "logsoftmax"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MaxPool"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "MaxPool3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MaxPool3DGrad"
  input_arg {
    name: "orig_input"
    type_attr: "TInput"
  }
  input_arg {
    name: "orig_output"
    type_attr: "TInput"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "TInput"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MaxPool3DGradGrad"
  input_arg {
    name: "orig_input"
    type_attr: "T"
  }
  input_arg {
    name: "orig_output"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NDHWC"
    }
    allowed_values {
      list {
        s: "NDHWC"
        s: "NCDHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MaxPoolGrad"
  input_arg {
    name: "orig_input"
    type_attr: "T"
  }
  input_arg {
    name: "orig_output"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPoolGradGrad"
  input_arg {
    name: "orig_input"
    type_attr: "T"
  }
  input_arg {
    name: "orig_output"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPoolGradGradWithArgmax"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "argmax"
    type_attr: "Targmax"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "Targmax"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPoolGradWithArgmax"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "argmax"
    type_attr: "Targmax"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "Targmax"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPoolWithArgmax"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "argmax"
    type_attr: "Targmax"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "Targmax"
    type: "type"
    default_value {
      type: DT_INT64
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "QuantizedAvgPool"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "ksize"
    type: "list(int)"
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedBatchNormWithGlobalNormalization"
  input_arg {
    name: "t"
    type_attr: "Tinput"
  }
  input_arg {
    name: "t_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "t_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "m"
    type_attr: "Tinput"
  }
  input_arg {
    name: "m_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "m_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "v"
    type_attr: "Tinput"
  }
  input_arg {
    name: "v_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "v_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "beta"
    type_attr: "Tinput"
  }
  input_arg {
    name: "beta_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "beta_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "gamma"
    type_attr: "Tinput"
  }
  input_arg {
    name: "gamma_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "gamma_max"
    type: DT_FLOAT
  }
  output_arg {
    name: "result"
    type_attr: "out_type"
  }
  output_arg {
    name: "result_min"
    type: DT_FLOAT
  }
  output_arg {
    name: "result_max"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "variance_epsilon"
    type: "float"
  }
  attr {
    name: "scale_after_normalization"
    type: "bool"
  }
}
op {
  name: "QuantizedBiasAdd"
  input_arg {
    name: "input"
    type_attr: "T1"
  }
  input_arg {
    name: "bias"
    type_attr: "T2"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_bias"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_bias"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_out"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_out"
    type: DT_FLOAT
  }
  attr {
    name: "T1"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "T2"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedConv2D"
  input_arg {
    name: "input"
    type_attr: "Tinput"
  }
  input_arg {
    name: "filter"
    type_attr: "Tfilter"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_filter"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_filter"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "Tfilter"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QINT32
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedMaxPool"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "ksize"
    type: "list(int)"
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedRelu"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedRelu6"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedReluX"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "max_value"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "Relu"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Relu6"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Relu6Grad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "ReluGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Softmax"
  input_arg {
    name: "logits"
    type_attr: "T"
  }
  output_arg {
    name: "softmax"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SoftmaxCrossEntropyWithLogits"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  input_arg {
    name: "labels"
    type_attr: "T"
  }
  output_arg {
    name: "loss"
    type_attr: "T"
  }
  output_arg {
    name: "backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Softplus"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SoftplusGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Softsign"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SoftsignGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SparseSoftmaxCrossEntropyWithLogits"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  input_arg {
    name: "labels"
    type_attr: "Tlabels"
  }
  output_arg {
    name: "loss"
    type_attr: "T"
  }
  output_arg {
    name: "backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "Tlabels"
    type: "type"
    default_value {
      type: DT_INT64
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "TopK"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  attr {
    name: "k"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "sorted"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  deprecation {
    version: 7
    explanation: "Use TopKV2 instead"
  }
}
op {
  name: "TopKV2"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "k"
    type: DT_INT32
  }
  output_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  attr {
    name: "sorted"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
