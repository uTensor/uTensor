#ifndef UTENSOR_S_QUANTIZED_FC_OPS_H
#define UTENSOR_S_QUANTIZED_FC_OPS_H
#include "context.hpp"
#include "operatorBase.hpp"

namespace uTensor {

DECLARE_ERROR(InvalidFCQuantizationScalesError);
DECLARE_ERROR(FCQuantizationScaleMultipleLTzeroError);

// Following two functions are borrowed from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/fully_connected.cc
// for operator compatibility
void GetQuantizedConvolutionMultipler(const Tensor& input, const Tensor& filter,
                                      const Tensor& bias, Tensor& output,
                                      double* multiplier);

void GetQuantizedConvolutionMultipler(const Tensor& input, const Tensor& filter,
                                      Tensor& output, double* multiplier);

}  // namespace uTensor
#endif
