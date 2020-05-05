#include "fully_connected_kernel.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
namespace TFLM {

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max) {

  int32_t input_offset = -input->get_quantization_params().get_zeroP_for_channel(0);
  int32_t filter_offset = -filter->get_quantization_params().get_zeroP_for_channel(0);
  int32_t output_offset = output->get_quantization_params().get_zeroP_for_channel(0);
  output_shift = -output_shift;
  //quantized_activation_min = data->output_activation_min;
  //quantized_activation_max = data->output_activation_max;

  const TensorShape& input_shape = input->get_shape();
  const TensorShape& filter_shape  = filter->get_shape();
  TensorShape& output_shape  = output->get_shape();

  const int filter_dim_count = filter_shape.num_dims();
  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  if(!(output_depth < filter_shape[filter_dim_count - 2])){
      Context::get_default_context()->throwError(
          new InvalidMatrixMultIndicesError);
  }
  const int accum_depth = filter_shape[filter_dim_count - 1];
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        // TODO write this in tensor form
        int32_t input_val = static_cast<int8_t>(input(b * accum_depth + d));
        int32_t filter_val = static_cast<int8_t>(filter(out_c * accum_depth + d));
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      //output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
      //output_data(out_c + output_depth * b) = static_cast<int8_t>(acc);
      output(b, out_c) = static_cast<int8_t>(acc);
    }
  }
}
void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, const Tensor& bias,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max) {

  int32_t input_offset = -input->get_quantization_params().get_zeroP_for_channel(0);
  int32_t filter_offset = -filter->get_quantization_params().get_zeroP_for_channel(0);
  int32_t output_offset = output->get_quantization_params().get_zeroP_for_channel(0);
  output_shift = -output_shift;
  //quantized_activation_min = data->output_activation_min;
  //quantized_activation_max = data->output_activation_max;

  const TensorShape& input_shape = input->get_shape();
  const TensorShape& filter_shape  = filter->get_shape();
  TensorShape& output_shape  = output->get_shape();

  const int filter_dim_count = filter_shape.num_dims();
  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  if(!(output_depth < filter_shape[filter_dim_count - 2])){
      Context::get_default_context()->throwError(
          new InvalidMatrixMultIndicesError);
  }
  const int accum_depth = filter_shape[filter_dim_count - 1];
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        // TODO write this in tensor form
        int32_t input_val = static_cast<int8_t>(input(b * accum_depth + d));
        int32_t filter_val = static_cast<int8_t>(filter(out_c * accum_depth + d));
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      acc += static_cast<int8_t>(bias(out_c));
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_activation_min);
      acc = std::min(acc, output_activation_max);
      //output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
      //output_data(out_c + output_depth * b) = static_cast<int8_t>(acc);
      output(b, out_c) = static_cast<int8_t>(acc);
    }
  }
}

}// TFLM
}// uTensor
