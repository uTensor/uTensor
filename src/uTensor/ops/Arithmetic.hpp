#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include <algorithm>
#include <limits>

#include "Arithmetic_kernels.hpp"
#include "uTensor/core/operatorBase.hpp"

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

template <typename T>
class SubOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : int8_t { a, b };
  enum names_out : int8_t { c };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    sub_kernel<T>(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

template <typename T>
class MulOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    mul_kernel<T>(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

template <typename T>
class DivOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };

 protected:
  virtual void compute() {
    div_kernel<T>(outputs[c].tensor(), inputs[a].tensor(), inputs[b].tensor());
  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor
#endif
