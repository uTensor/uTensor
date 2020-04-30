#ifndef UTENSOR_S_QUANTIZE_UTILS_H
#define UTENSOR_S_QUANTIZE_UTILS_H
#include <cmath>
#include <limits>
#include "context.hpp"
#include "tensor.hpp"
#include "tflm_defs.hpp"

namespace uTensor {
namespace TFLM {

DECLARE_ERROR(SymmetricQuantizationFixedPointError);
DECLARE_ERROR(SymmetricQuantizationFixedPointRangeError);

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

}
}

#endif
