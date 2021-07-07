#include "QuantizeOps.hpp"

namespace uTensor {
namespace TflmSymQuantOps {

// dummy specification
template <>
void dequantize_kernel<float, float>(Tensor& b, const Tensor& a) {
  const int flat_size = b->get_shape().get_linear_size();
  for (int i = 0; i < flat_size; i++) {
    b(i) = static_cast<float>(a(i));
  }
}

}  // namespace TflmSymQuantOps
}  // namespace uTensor