#include "ActivationFncs.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <>
void SoftmaxOperator<int8_t>::compute() {
  const Tensor& inT = inputs[in].tensor();
  Tensor& outT = outputs[out].tensor();
  // TODO Check sizes here and throw mismatch
  uint32_t in_size = inT->get_shape().get_linear_size();
  uint32_t out_size = outT->get_shape().get_linear_size();
  if (in_size != out_size)
    Context::get_default_context()->throwError(
        new OperatorIOSizeMismatchError);
  sq_softmax_k(outT, inT, beta);
}


} // ReferenceOperators
} // uTensor
