#ifndef UTENSOR_STRIDED_SLICE_KERNELS_OPS_H
#define UTENSOR_STRIDED_SLICE_KERNELS_OPS_H

#include "context.hpp"
#include "operatorBase.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "uTensor_util.hpp"

namespace uTensor {
namespace ReferenceOperators {

void ValidateStridedSliceOp(
    const Tensor& input, const Tensor& begin_tensor, const Tensor& end_tensor,
    const Tensor& strides_tensor, int32_t begin_mask_spec,
    int32_t end_mask_spec, const int32_t ellipsis_mask, int32_t new_axis_mask,
    int32_t shrink_axis_mask
    /*
    TF optimization for StridedSliceOp is not supported yet, following
    aruguments are not used
    */
    // TensorShape& processing_shape,
    // TensorShape& final_shape,
    // bool& is_identity, bool& is_simple_slice, bool& slice_dim0,
    // std::array<int64_t, 4>& begin, std::array<int64_t, 4> &end,
    // std::array<int64_t, 4>& strides, StridedSliceShapeSpec* shape_spec
);

/*
 * Requires proper testing, tested only for some cases
 * Not supported features: Elipses_axis mask, new_axis_mask, undefined dimension
 * size (dim == -1)
 */
template <typename T>
void stridedslice_kernel(const Tensor& input, const Tensor& begin_tensor,
                         const Tensor& end_tensor, const Tensor& strides_tensor,
                         Tensor& output, int begin_mask, int end_mask,
                         int ellipsis_mask, int new_axis_mask,
                         int shrink_axis_mask) {
  /*
    - https://www.tensorflow.org/api_docs/python/tf/strided_slice
      -
    https://github.com/tensorflow/tensorflow/blob/972943258dda58324dafa53b1ecba95d40677055/tensorflow/core/kernels/strided_slice_op.cc
      -
    https://github.com/tensorflow/tensorflow/blob/e8d65b3dd3e96e543177f7334a1e865b2d3c8f2e/tensorflow/core/util/strided_slice_op.cc
  */
  ValidateStridedSliceOp(input, begin_tensor, end_tensor, strides_tensor,
                         begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                         shrink_axis_mask);
  size_t ndims = begin_tensor.get_shape().num_dims();
  StridedIterator stride_it(input, begin_tensor, end_tensor, strides_tensor,
                            begin_mask, end_mask);
  for (uint32_t i = 0; i < output.get_shape().num_elems(); ++i) {
    int32_t src_idx = stride_it.next();
    if (src_idx < 0) {
      uTensor_printf("inconsistent output size and strided slice");
      Context::get_default_context()->throwError(new InvalidTensorOutputError);
      return;
    }
    T value = input(src_idx);
    output(i) = value;
  }
}
}  // namespace ReferenceOperators
}  // namespace uTensor

#endif  // UTENSOR_STRIDED_SLICE_KERNELS_OPS_H
