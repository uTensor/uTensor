#include "operatorBase.hpp"
#include "tensor.hpp"

namespace uTensor {
namespace ReferenceOperators {
// https://github.com/tensorflow/tensorflow/blob/12a806e96866296b154134b27ef4228f39f403cc/tensorflow/lite/micro/kernels/reduce.cc#L83
class ReduceOperator : public OperatorInterface<1, 1> {
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
  void compute() {
    Tensor& input_tensor = inputs[input].tensor();
    Tensor& output_tensor = outputs[output].tensor();
    for (uint32_t i = 0; i < output_tensor->num_elems(); ++i) {
      output_tensor(i) = static_cast<T>(0);
    }
    T denum = 1;
    for (auto d : _dims) {
      denum *= input_tensor->get_shape()[d];
    }
    for (uint32_t offset = 0; offset < input_tensor->num_elems(); ++offset) {
      T value = static_cast<T>(input_tensor(offset));
      uint32_t new_offset = adjust_linear_idx(input_tensor, offset);
      output_tensor(new_offset) =
          static_cast<T>(output_tensor(new_offset)) + value;
    }
    for (uint32_t offset = 0; offset < output_tensor->num_elems(); ++offset) {
      T total_value = static_cast<T>(output_tensor(offset));
      output_tensor(offset) = total_value / denum;
    }
  }
};
}  // namespace ReferenceOperators
}  // namespace uTensor