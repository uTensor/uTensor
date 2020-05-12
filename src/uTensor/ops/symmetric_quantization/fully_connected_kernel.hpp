#ifndef UTENSOR_S_QUANTIZED_FC_KERNELS_H
#define UTENSOR_S_QUANTIZED_FC_KERNELS_H
#include "Matrix.hpp"
#include "context.hpp"
#include "operatorBase.hpp"

namespace uTensor {
namespace TFLM {

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max);

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, const Tensor& bias,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max);

}  // namespace TFLM

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, const Tensor& bias,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max);
}  // namespace uTensor

#endif
