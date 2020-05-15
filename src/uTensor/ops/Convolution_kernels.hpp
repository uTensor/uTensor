#ifndef UTENSOR_CONVOLUTION_KERNELS_H
#define UTENSOR_CONVOLUTION_KERNELS_H
#include <algorithm>
#include <limits>

#include "operatorBase.hpp"

namespace uTensor {

enum Padding : uint8_t { UNKNOWN = 0, VALID = 1, SAME = 2 };

template <typename T, typename Filter>
void generic_convolution_kernel(Tensor& out, const Tensor& in, Filter filter,
                                const Padding padding,
                                const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = filter.out_channels();
  const int16_t filter_rows = filter.height();
  const int16_t filter_cols = filter.width();
  const int16_t filter_count = filter.out_channels();

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
          // T output_val = 0;
          filter.reset();
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
                filter.PartialCompute(input_value, filter_y, filter_x,
                                      in_channel, out_channel);
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = filter.finalize();
        }
      }
    }
  }
}
// Hack until I get the generic version working
template <typename T, typename Filter>
void generic_pool_convolution_kernel(Tensor& out, const Tensor& in,
                                     Filter filter, const Padding padding,
                                     const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = input_depth;  // filter.out_channels();
  const int16_t filter_rows = filter.height();
  const int16_t filter_cols = filter.width();
  // const int16_t filter_count = filter.out_channels();

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
    // filter_left_offset =
    //    ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    // filter_top_offset =
    //    ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
    filter_left_offset =
        (((input_cols - filter_cols) / stride_cols) * stride_cols +
         filter_cols - input_cols + 1) /
        2;
    filter_top_offset =
        (((input_rows - filter_rows) / stride_rows) * stride_rows +
         filter_rows - input_rows + 1) /
        2;
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
    // Each filter kernel produces one output channel.
    for (int out_channel = 0; out_channel < out_depth;
         ++out_channel) {  // Thank god for no caches
      for (int out_y = 0; out_y < out_rows; ++out_y) {
        for (int out_x = 0; out_x < out_cols; ++out_x) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          // T output_val = 0;
          filter.reset();
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              // for (int in_channel = 0; in_channel < input_depth;
              // ++in_channel) {
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
                input_value = in(batch, in_y, in_x, out_channel);
              } else {
                input_value = 0;
              }
              // size_t filter_index = filter_y * filter_cols * input_depth *
              // filter_count +
              //    filter_x * input_depth * filter_count +
              //    in_channel * filter_count + out_channel;
              // const T filter_value = filter(filter_index);
              filter.PartialCompute(input_value, filter_y, filter_x,
                                    out_channel, out_channel);
              //}
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = filter.finalize();
        }
      }
    }
  }
}

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
void depthwise_separable_convolution_kernel(Tensor& out, const Tensor& in,
                                            const Tensor& dw_filter,
                                            const Tensor& pw_filter,
                                            const Padding padding,
                                            const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& df_shape = dw_filter->get_shape();
  const TensorShape& pf_shape = pw_filter->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = pf_shape[3];
  const int16_t dw_filter_rows = df_shape[0];
  const int16_t dw_filter_cols = df_shape[1];
  const int16_t dw_filter_in_channels = df_shape[2];
  const int16_t dw_filter_channel_mult = df_shape[3];
  const int16_t pw_filter_in_channels = pf_shape[2];

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
        ((out_cols - 1) * stride_cols + dw_filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + dw_filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + dw_filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + dw_filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Fuse the Depthwise filtering with the pointwise filtering by
        // iterating over the channels first
        for (int out_channel = 0; out_channel < out_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          T output_val = 0;

          for (int filter_y = 0; filter_y < dw_filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < dw_filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                for (int r = 0; r < dw_filter_channel_mult; ++r) {
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
                  // size_t filter_index = filter_y * filter_cols * input_depth
                  // * filter_count +
                  //    filter_x * input_depth * filter_count +
                  //    in_channel * filter_count + out_channel;
                  // const T filter_value = filter(filter_index);
                  const T dw_filter_value =
                      dw_filter(filter_y, filter_x, in_channel, r);
                  const T pw_filter_value =
                      pw_filter(0, 0, in_channel * dw_filter_channel_mult + r,
                                out_channel);
                  output_val +=
                      (input_value * dw_filter_value * pw_filter_value);
                }
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


void uComputePaddingHeightWidth(int stride_height, int stride_width,
                               int dilation_rate_height,
                               int dilation_rate_width, int in_height,
                               int in_width, int filter_height,
                               int filter_width, int* padding_height,
                               int* padding_width, Padding padding,
                               int* out_height, int* out_width);

template <typename T>
void depthwise_separable_convolution_kernel_v2(Tensor& output, const Tensor& input,
                                            const Tensor& filter,
                                            const Tensor& bias,
                                            const Padding padding,
                                            const uint16_t (&strides)[4],
                                            const int depth_multiplier,
                                            const uint16_t (&dialation)[2]
                                            ) {
  
  // Check dimensions of the tensors.
  const TensorShape& input_shape = input->get_shape();
  const TensorShape& filter_shape = filter->get_shape();
  const TensorShape& output_shape = output->get_shape();
  
  const int channels_out = filter_shape[3];
  const int batches = input_shape[0];
  const int output_depth = output_shape[3]; // This should be the same as filter_shape[3]
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int input_width = input_shape[2];
  const int input_height = input_shape[1];
  const int input_depth = input_shape[3];
  const int filter_width = filter_shape[2];
  const int filter_height = filter_shape[1];
  const int stride_width = strides[2];
  const int stride_height = strides[1];
  const int dialation_width_factor = dialation[1];
  const int dialation_height_factor = dialation[0];

  int unused_output_height, unused_output_width;
  int32_t pad_width, pad_height;

  uComputePaddingHeightWidth(stride_height, stride_width, 1, 1, input_height,
                                  input_width, filter_height, filter_width,
                                  &pad_height, &pad_width,
                                  padding,
                                  &unused_output_height, &unused_output_width);
  
  if (!(input_shape.num_dims() == 4)) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (!(filter_shape.num_dims() == 4)) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (!(output_shape.num_dims() == 4)) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (!(output_depth == filter_shape[3])) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (!(batches == output_shape[0])) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dialation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dialation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  // int32_t input_val = input_data[Offset(input_shape, batch,
                  // in_y,
                  //                                     in_x, in_channel)];
                  T input_val =
                      static_cast<T>(input(batch, in_y, in_x, in_channel));
                  // int32_t filter_val = filter_data[Offset(
                  //     filter_shape, 0, filter_y, filter_x, output_channel)];
                  T filter_val = static_cast<T>(
                      filter(filter_y, filter_x, output_channel));
                  acc += filter_val * (input_val);
                }
              }
            }
            // assuming bias data will always be provided
            acc += static_cast<T>(bias(output_channel));

            output(batch, out_y, out_x, output_channel) =
                static_cast<T>(acc);
          }
        }
      }
    }
  }
}

}  // namespace uTensor
#endif
