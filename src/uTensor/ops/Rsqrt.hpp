#ifndef UTENSOR_RSQRT_H
#define UTENSOR_RSQRT_H
#include <cmath>

#include "uTensor/core/operatorBase.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/types.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename Tin>
class RsqrtOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };

 protected:
  void compute() {
    Tensor &inputT = inputs[input].tensor();
    Tensor &outputT = outputs[output].tensor();
    for (uint32_t i = 0; i < outputT->num_elems(); ++i) {
      Tin v = static_cast<Tin>(inputT(i));
      Tin one = 1;
      Tin sqrt = std::sqrt(v);
      outputT(i) = static_cast<Tin>(one / sqrt);
    }
  }
};
}  // namespace ReferenceOperators
}  // namespace uTensor
#endif  // UTENSOR_RSQRT_H