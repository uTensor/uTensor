#ifndef UTENSOR_S_QUANTIZE_UTILS_H
#define UTENSOR_S_QUANTIZE_UTILS_H
#include <cmath>
#include <limits>

#include "uTensor/core/context.hpp"
#include "uTensor/core/tensor.hpp"
#include "tflm_defs.hpp"

namespace uTensor {

DECLARE_ERROR(SymmetricQuantizationFixedPointError);
DECLARE_ERROR(SymmetricQuantizationFixedPointRangeError);
DECLARE_ERROR(InvalidFCQuantizationScalesError);
DECLARE_ERROR(FCQuantizationScaleMultipleLTzeroError);

namespace TFLM {

typedef struct TfLiteAffineQuantization {
  float* scale;
  int32_t* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;

int64_t IntegerFrExp(double input, int* shift);

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift);

void CalculateActivationRangeQuantizedImpl(TfLiteFusedActivation activation,
                                           int32_t qmin, int32_t qmax,
                                           Tensor& output, int32_t* act_min,
                                           int32_t* act_max);

void CalculateActivationRangeQuantized(TfLiteFusedActivation activation,
                                       Tensor& output, int32_t* act_min,
                                       int32_t* act_max);

// Following two functions are borrowed from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/fully_connected.cc
// for operator compatibility
void GetQuantizedConvolutionMultipler(const Tensor& input, const Tensor& filter,
                                      const Tensor& bias, Tensor& output,
                                      double* multiplier);

void GetQuantizedConvolutionMultipler(const Tensor& input, const Tensor& filter,
                                      Tensor& output, double* multiplier);

int32_t MultiplyByQuantizedMultiplier(int32_t acc, int32_t output_multiplier,
                                      int32_t output_shift);

}  // namespace TFLM
}  // namespace uTensor

#endif
