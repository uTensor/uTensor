#ifndef UTENSOR_FUNCTIONAL_OPS_HPP
#define UTENSOR_FUNCTIONAL_OPS_HPP

#include <algorithm>
#include <limits>
#include <vector>

#include "functional_kernels.hpp"
#include "operatorBase.hpp"
namespace uTensor {
namespace ReferenceOperators {

class InPlaceFnc : public OperatorInterface<1, 0> {
 public:
  enum names_in : uint8_t { x };

 protected:
  virtual void compute() = 0;
};

template <typename T>
class MinOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    min_kernel<T>(outputs[out].tensor(), inputs[in].tensor());
  }
};

template <typename T>
class MaxOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

 protected:
  virtual void compute() {
    max_kernel<T>(outputs[out].tensor(), inputs[in].tensor());
  }
};


template <typename T>
class SqueezeOperator : public InPlaceFnc {
 public:
   SqueezeOperator() : _axis() {}
   SqueezeOperator(std::initializer_list<uint8_t> axis) : _axis(axis) {}

 protected:
  virtual void compute() { squeeze_kernel<T>(inputs[x].tensor(), _axis); }
 private:
  std::vector<uint8_t> _axis;
};
}
}

#endif
