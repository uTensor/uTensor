#ifndef UTENSOR_S_QUANTIZED_FC_KERNELS_H
#define UTENSOR_S_QUANTIZED_FC_KERNELS_H
#include "Matrix.hpp"
#include "context.hpp"
#include "operatorBase.hpp"
#include <functional>

namespace uTensor {
namespace TFLM {

void quantized_matrix_mult_kernel(Tensor& output, const Tensor& input,
                                  const Tensor& filter, std::function<int32_t(int32_t)> bias,
                                  int32_t output_multiplier, int output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max);

}  // namespace TFLM
}  // namespace uTensor

#endif
