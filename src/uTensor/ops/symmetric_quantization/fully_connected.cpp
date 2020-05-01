#include "fully_connected.hpp"

namespace uTensor {
DECLARE_ERROR(InvalidFCQuantizationScalesError);
DECLARE_ERROR(FCQuantizationScaleMultipleLTzeroError);

// Following two functions are borrowed from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/fully_connected.cc for operator compatibility
void GetQuantizedConvolutionMultipler(const Tensor& input,
                                      const Tensor& filter,
                                      const Tensor& bias,
                                      Tensor& output,
                                      double* multiplier) {
  const double input_product_scale = static_cast<double>(input->get_quantization_params().get_scale_for_channel(0)) *
                                     static_cast<double>(filter->get_quantization_params().get_scale_for_channel(0));
  // TODO(ahentz): The following conditions must be guaranteed by the training
  // pipeline.
  if (bias) {
    const double bias_scale = static_cast<double>(bias->get_quantization_params().get_scale_for_channel(0));
      if(!(std::abs(input_product_scale - bias_scale) <=
                       1e-6 * std::min(input_product_scale, bias_scale))){
        Context::get_default_context()->throwError( new InvalidFCQuantizationScalesError );
      }
  }
  return GetQuantizedConvolutionMultipler(input, filter, output,
                                          multiplier);
}

void GetQuantizedConvolutionMultipler(const Tensor& input,
                                      const Tensor& filter,
                                      Tensor& output,
                                      double* multiplier) {
  const double input_product_scale =
      static_cast<double>(input->get_quantization_params().get_scale_for_channel(0) * filter->get_quantization_params().get_scale_for_channel(0));
  if(!(input_product_scale >= 0)){
    Context::get_default_context()->throwError( new FCQuantizationScaleMultipleLTzeroError );
  }
  *multiplier = input_product_scale / static_cast<double>(output->get_quantization_params().get_scale_for_channel(0));

  return;
}

}
