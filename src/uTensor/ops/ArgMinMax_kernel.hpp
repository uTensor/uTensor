#ifndef UTENSOR_ARG_MIN_MAX_KERNEL_H
#define UTENSOR_ARG_MIN_MAX_KERNEL_H

#include "uTensor/core/context.hpp"
#include "uTensor/core/types.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensor_util.hpp"

namespace uTensor {
enum ArgMinMaxCompareFlag {
  Max = 1,
  Min = -1,
};

template <typename Tin>
void arg_min_max_kernel(Tensor& output, const Tensor& input, const Tensor& axis, ArgMinMaxCompareFlag cmp = Max) {
  if (axis->get_type() != u32) {
    // TODO: type convertion to u32?
    uTensor_printf("only support u32 typed axis tensor\n");
    Context::get_default_context()->throwError(new InvalidTensorError);
    return;
  }
  if (axis.get_shape().num_dims() != 1) {
    uTensor_printf("axis should be scalar\n");
    Context::get_default_context()->throwError(new InvalidTensorError);
    return;
  }
  if (output->get_type() != u32) {
    uTensor_printf("output type must be u32\n");
    Context::get_default_context()->throwError(new InvalidTensorError);
    return;
  }
  uint32_t axis_dim = axis(0);
  TensorShape input_shape = input->get_shape();
  int num_dims = input_shape.num_dims();
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
      Tin min_max_value = input(outer * axis_size * inner_size + inner);
      uint32_t min_max_index = 0;
      for (uint32_t i = 1; i < axis_size; ++i) {
        const Tin curr_value = input((outer * axis_size + i) * inner_size + inner);
        if (cmp * curr_value >= cmp * min_max_value) {
          min_max_value = curr_value;
          min_max_index = i;
        }
      }
      output(outer * inner_size + inner) = min_max_index;
    }
  }
}
} // namespace uTensor

#endif // UTENSOR_ARG_MIN_MAX_KERNEL_H
