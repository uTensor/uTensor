#include "trigonometric.hpp"

namespace uTensor {
namespace ReferenceOperators {

void TanhOperator<float, float>::compute() {
  Tensor& in = inputs[act_in].tensor();
  Tensor& out = outputs[act_out].tensor();

  const uint32_t flat_size = in->get_shape().get_linear_size();

  for (uint32_t i = 0; i < flat_size; i++) {
    const float in_val = in(i);
    float element_activation = tanh(in_val);
    out(i) = element_activation;
  }
}

void SinOperator<float, float>::compute() {
  Tensor& in = inputs[act_in].tensor();
  Tensor& out = outputs[act_out].tensor();

  const uint32_t flat_size = in->get_shape().get_linear_size();

  for (uint32_t i = 0; i < flat_size; i++) {
    const float in_val = in(i);
    float element_activation = sin(in_val);
    out(i) = element_activation;
  }
}

}  // namespace ReferenceOperators
}  // namespace uTensor