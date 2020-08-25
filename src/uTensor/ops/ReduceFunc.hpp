#include "operatorBase.hpp"
#include "tensor.hpp"

namespace uTensor {
namespace ReferenceOperators {
// https://github.com/tensorflow/tensorflow/blob/12a806e96866296b154134b27ef4228f39f403cc/tensorflow/lite/micro/kernels/reduce.cc#L83
class ReduceOperator : public OperatorInterface<2, 1> {
 public:
  ReduceOperator(initializer_list<uint16_t> dims);
  uint32_t adjust_linear_idx(Tensor& tensor, uint32_t idx);

 private:
  uint16_t _dims[4];
};

template <typename T>
class ReduceMeanOperator : ReduceOperator {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };
  ReduceMeanOperator(initializer_list<uint32_t> dims) : ReduceOperator(dims) {}

 protected:
  void compute() {
    Tensor& input = inputs[input].tensor();
    Tensor& output = outputs[output].tensor();
    for (uint32_t i = 0; i < output->num_elems(); ++i) {
      output(i) = static_cast<T>(0);
    }
    T denum = 1;
    for (auto d : _dims) {
      denum *= input->get_shape()[d];
    }
    for (uint32_t offset = 0; offset < input->num_elems(); ++offset) {
      uint32_t new_offset = adjust_linear_idx(input, offset);
      T value = input(offset) / denum;
      output[new_offset] += value;
    }
  }
};
}  // namespace ReferenceOperators
}  // namespace uTensor