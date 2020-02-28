#ifndef UTENSOR_CONVOLUTION_OPS_H
#define UTENSOR_CONVOLUTION_OPS_H
#include "operatorBase.hpp"

namespace uTensor {

enum Padding : uint8_t { VALID = 1, SAME = 2 };

template <typename T>
void convolution_kernel(Tensor& out, const Tensor& in, const Tensor& filter,
                        const Padding padding, const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& f_shape = filter->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = f_shape[3];
  const int16_t filter_rows = f_shape[0];
  const int16_t filter_cols = f_shape[1];
  const int16_t filter_count = f_shape[3];

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Each filter kernel produces one output channel.
        for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          T output_val = 0;
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                T input_value;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  input_value = in(batch, in_y, in_x, in_channel);
                } else {
                  input_value = 0;
                }
                // size_t filter_index = filter_y * filter_cols * input_depth *
                // filter_count +
                //    filter_x * input_depth * filter_count +
                //    in_channel * filter_count + out_channel;
                // const T filter_value = filter(filter_index);
                const T filter_value =
                    filter(filter_y, filter_x, in_channel, out_channel);
                output_val += (input_value * filter_value);
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = output_val;
        }
      }
    }
  }
}

template <typename T>
class ConvOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { in, filter };
  enum names_out : uint8_t { out };
  ConvOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    convolution_kernel<T>(*outputs[out].tensor, *inputs[in].tensor,
                          *inputs[filter].tensor, _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
void depthwise_seperable_convolution_kernel(Tensor& out, const Tensor& in,
                                            const Tensor& filter,
                                            const Padding padding,
                                            const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& f_shape = filter->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = f_shape[3];
  const int16_t filter_rows = f_shape[0];
  const int16_t filter_cols = f_shape[1];
  const int16_t filter_count = f_shape[3];

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Each filter kernel produces one output channel.
        for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          T output_val = 0;
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                T input_value;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  input_value = in(batch, in_y, in_x, in_channel);
                } else {
                  input_value = 0;
                }
                // size_t filter_index = filter_y * filter_cols * input_depth *
                // filter_count +
                //    filter_x * input_depth * filter_count +
                //    in_channel * filter_count + out_channel;
                // const T filter_value = filter(filter_index);
                const T filter_value =
                    filter(filter_y, filter_x, in_channel, out_channel);
                output_val += (input_value * filter_value);
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = output_val;
        }
      }
    }
  }
}

template <typename T>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, depthwise_filter, pointwise_filter };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  DepthwiseSeparableConvOperator(std::initializer_list<uint16_t> strides,
                                 Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape = *inputs[in].tensor->get_shape();
    TensorShape& df_shape = *inputs[depthwise_filter].tensor->get_shape();
    TensorShape& pf_shape = *inputs[pointwise_filter].tensor->get_shape();
    TensorShape& out_shape = *outputs[out].tensor->get_shape();

    convolution_kernel<T>(*outputs[out].tensor, *inputs[in].tensor,
                          *inputs[depthwise_filter].tensor, _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

}  // namespace uTensor
#endif
