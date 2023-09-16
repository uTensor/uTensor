#ifndef UTENSOR_SUM_H
#define UTENSOR_SUM_H

#include "uTensor/core/operatorBase.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/types.hpp"

namespace uTensor {
namespace ReferenceOperators {
template <typename Tin>
class SumOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { input, axis };
  enum names_out : uint8_t { output };

  virtual void compute() {
    const Tensor &input_tensor = inputs[input].tensor();
    const Tensor &axis_tensor = inputs[axis].tensor();
    Tensor &output_tensor = outputs[output].tensor();
    if (axis_tensor->get_type() != i32) {
      uTensor_printf("only support i32 typed axis tensor\n");
      Context::get_default_context()->throwError(new InvalidTensorError);
      return;
    }
    TensorShape input_shape = input_tensor->get_shape();
    int num_dims = input_shape.num_dims();
    int32_t axis_dim = axis_tensor(0);
    if (axis_dim < 0) axis_dim += num_dims;
    uint16_t axis_size = input_shape[axis_dim];
    uint32_t outer_size = 1;
    for (int i = 0; i < axis_dim; ++i) {
      outer_size *= input_shape[i];
    }
    uint32_t inner_size = 1;
    for (int i = axis_dim + 1; i < num_dims; ++i) {
      inner_size *= input_shape[i];
    }
    for (uint32_t outer = 0; outer < outer_size; ++outer) {
      for (uint32_t inner = 0; inner < inner_size; ++inner) {
        Tin acc = 0;
        for (uint32_t i = 1; i < axis_size; ++i) {
          Tin elem = input_tensor((outer * axis_size + i) * inner_size + inner);
          acc += elem;
        }
        output_tensor(outer * inner_size + inner) = acc;
      }
    }
  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor

#endif  // UTENSOR_SUM_H