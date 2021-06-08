#ifndef UTENSOR_CONCAT_KERNELS_H
#define UTENSOR_CONCAT_KERNELS_H

#include "uTensor/core/operatorBase.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename T>
void concat_kernel(const Tensor &a, const Tensor &b, int32_t axis_idx,
                   Tensor &c) {
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  TensorShape c_shape = c->get_shape();
  uint32_t num_dims = a_shape.num_dims();
  if (axis_idx < 0) {
    axis_idx += num_dims;
  }
  // sanity checks
  if (num_dims != b_shape.num_dims()) {
    uTensor_printf(
        "[Error] input tensors must have the sampe number of dimensions\n");
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
    return;
  }
  if (num_dims != c_shape.num_dims()) {
    uTensor_printf("[Error] invalid output tensor dimensions\n");
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (c_shape[axis_idx] != a_shape[axis_idx] + b_shape[axis_idx]) {
    uTensor_printf("[Error] invalid output shape:");
    c_shape.print(true);
    Context::get_default_context()->throwError(new InvalidTensorOutputError);
    return;
  }
  bool shape_mismatch = false;
  for (uint32_t i = 0; i < num_dims; ++i) {
    if (i != axis_idx && a_shape[i] != b_shape[i]) {
      shape_mismatch = true;
    }
  }
  if (shape_mismatch) {
    uTensor_printf("[Error] input tensors shape mismatch: ");
    a_shape.print();
    uTensor_printf(" and ");
    b_shape.print(true);
    Context::get_default_context()->throwError(new InvalidTensorInputError);
    return;
  }
  // end of sanity checks

  uTensor_printf("Start my_concat_kernel\n");
  uTensor_printf("a_shape: ");
  a_shape.print(true);
  uTensor_printf("b_shape: ");
  b_shape.print(true);
  uTensor_printf("c_shape: ");
  c_shape.print(true);

  uint32_t outer_size = 1;
  for (uint32_t i = 0; i < axis_idx; ++i) {
    outer_size *= c_shape[i];
  }
  uint32_t inner_size = 1;
  for (uint32_t i = axis_idx + 1; i < num_dims; ++i) {
    inner_size *= c_shape[i];
  }
  // copy a then b
  uint32_t a_axis_size = a_shape[axis_idx];
  uint32_t b_axis_size = b_shape[axis_idx];
  uint32_t out_axis_size = a_axis_size + b_axis_size;
  for (uint32_t outer = 0; outer < outer_size; ++outer) {
    for (uint32_t inner = 0; inner < inner_size; ++inner) {
      // copy a
      for (uint32_t i = 0; i < a_axis_size; ++i) {
        uint32_t src_offset = (outer * a_axis_size + i) * inner_size + inner;
        uint32_t dest_offset = (outer * out_axis_size + i) * inner_size + inner;
        c(dest_offset) = static_cast<T>(a(src_offset));
      }
      // copy b
      for (uint32_t i = 0; i < b_axis_size; ++i) {
        uint32_t src_offset = (outer * b_axis_size + i) * inner_size + inner;
        uint32_t dest_offset =
            (outer * out_axis_size + a_axis_size + i) * inner_size + inner;
        c(dest_offset) = static_cast<T>(b(src_offset));
      }
    }
  }
  return;
}

}  // namespace ReferenceOperators
}  // namespace uTensor

#endif  // UTENSOR_CONCAT_KERNELS_H