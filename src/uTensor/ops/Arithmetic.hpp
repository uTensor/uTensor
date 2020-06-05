#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include <algorithm>
#include <limits>

#include "Arithmetic_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename T>
class AddOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    add_kernel<T>(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

}
}  // namespace uTensor
#endif
