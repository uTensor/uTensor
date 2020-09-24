#include "operatorBase.hpp"
#include "tensor.hpp"

namespace uTensor {
namespace ReferenceOperators {
// https://github.com/tensorflow/tensorflow/blob/12a806e96866296b154134b27ef4228f39f403cc/tensorflow/lite/micro/kernels/reduce.cc#L83
class ReduceOperator : public OperatorInterface<2, 1> {
 public:
  ReduceOperator(initializer_list<uint16_t> dims);
  uint32_t adjust_linear_idx(Tensor& tensor, uint32_t idx);

 protected:
  uint16_t _dims[4];
};

template <typename T>
class ReduceMeanOperator : public ReduceOperator {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };
  ReduceMeanOperator(initializer_list<uint16_t> dims) : ReduceOperator(dims) {}

 protected:
  void compute();
};

template<typename T>
void ReduceMeanOperator<T>::compute() {
  Tensor& inputT = inputs[input].tensor();
  Tensor& outputT = outputs[output].tensor();
  for (uint32_t i = 0; i < outputT->num_elems(); ++i) {
    outputT(i) = static_cast<T>(0);
  }
  T denum = 1;
  for (auto d : _dims) {
    denum *= inputT->get_shape()[d];
  }
  for (uint32_t offset = 0; offset < inputT->num_elems(); ++offset) {
    uint32_t new_offset = adjust_linear_idx(inputT, offset);
    T value = static_cast<T>(inputT(offset)) / denum;
    outputT(new_offset) = static_cast<T>(outputT(new_offset)) + value;
  }
}

template <>
void ReduceMeanOperator<int8_t>::compute();

}  // namespace ReferenceOperators
}  // namespace uTensor
