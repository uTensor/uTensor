#include "depthwise_separable_convolution_kernels.hpp"

namespace uTensor {
namespace TFLM {

DEFINE_ERROR(qDwsMismatchedDimensionsError);
DEFINE_ERROR(InvalidQDwsActivationRangeError);
DEFINE_ERROR(InvalidQDwsOutputDepthError);

TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  TfLitePaddingValues padding_values;
  int offset = 0;
  padding_values.height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  padding_values.height_offset = offset;
  padding_values.width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  padding_values.width_offset = offset;
  return padding_values;
}

void ComputePaddingHeightWidth(int stride_height, int stride_width,
                               int dilation_rate_height,
                               int dilation_rate_width, int in_height,
                               int in_width, int filter_height,
                               int filter_width, int32_t* padding_height,
                               int32_t* padding_width, TfLitePadding padding,
                               int* out_height, int* out_width) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  //TfLitePaddingValues padding_values;
  int offset = 0;
  *padding_height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  // padding_values.height_offset = offset;
  *padding_width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  // padding_values.width_offset = offset;
  // return padding_values;
}

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1) {
  if (!(s0[i0] == s1[i1])) {
    Context::get_default_context()->throwError(
        new qDwsMismatchedDimensionsError);
  }

  return s0[i0];
}

void DepthwiseConvPerChannel(const DepthwiseParams& params,
                             const int32_t* output_multiplier,
                             const int32_t* output_shift, Tensor& input,
                             Tensor& filter, Tensor& bias, Tensor& output) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TensorShape input_shape = input->get_shape();
  TensorShape filter_shape = filter->get_shape();
  TensorShape output_shape = output->get_shape();

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

  if (!(output_activation_min < output_activation_max)) {
    Context::get_default_context()->throwError(
        new InvalidQDwsActivationRangeError);
  }
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const int filter_height = filter_shape[1];
  const int filter_width = filter_shape[2];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  if (!(output_depth == input_depth * depth_multiplier)) {
    Context::get_default_context()->throwError(new InvalidQDwsOutputDepthError);
  }
  if (!(bias->get_shape().get_linear_size() == (uint16_t)output_depth)) {
    Context::get_default_context()->throwError(new InvalidQDwsOutputDepthError);
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
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  // int32_t input_val = input_data[Offset(input_shape, batch,
                  // in_y,
                  //                                     in_x, in_channel)];
                  int32_t input_val =
                      static_cast<int8_t>(input(batch, in_y, in_x, in_channel));
                  // int32_t filter_val = filter_data[Offset(
                  //     filter_shape, 0, filter_y, filter_x, output_channel)];
                  int32_t filter_val = static_cast<int8_t>(
                      filter(0, filter_y, filter_x, output_channel));
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8, even though it
                  // is represented using int32.
                  // int32 += int8 * (int8 - int8) so the highest value we can
                  // get from each accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
            // assuming bias data will always be provided
            acc += static_cast<int32_t>(bias(output_channel));
            acc = MultiplyByQuantizedMultiplier(
                acc, output_multiplier[output_channel],
                output_shift[output_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output(batch, out_y, out_x, output_channel) =
                static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}

}  // namespace TFLM
}  // namespace uTensor
