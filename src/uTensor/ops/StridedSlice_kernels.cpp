#include "StridedSlice.hpp"

namespace uTensor {
namespace ReferenceOperators {
void ValidateStridedSliceOp(const Tensor& input, const Tensor& begin_tensor,
                            const Tensor& end_tensor,
                            const Tensor& strides_tensor,
                            int32_t begin_mask_spec, int32_t end_mask_spec,
                            const int32_t ellipsis_mask, int32_t new_axis_mask,
                            int32_t shrink_axis_mask) {
  const TensorShape& begin_tensor_shape = begin_tensor->get_shape();
  uint32_t begin_tensor_size = begin_tensor_shape.num_elems();

  const TensorShape& end_tensor_shape = end_tensor->get_shape();
  uint32_t end_tensor_size = end_tensor_shape.num_elems();

  const TensorShape& strides_tensor_shape = strides_tensor->get_shape();
  uint32_t strides_tensor_size = strides_tensor_shape.num_elems();

  if (begin_tensor_shape.num_dims() != 1 || end_tensor_shape.num_dims() != 1 ||
      strides_tensor_shape.num_dims() != 1 ||
      strides_tensor_size != begin_tensor_size ||
      strides_tensor_size != end_tensor_size ||
      strides_tensor_size != input->get_shape().num_dims()) {
    ERR_EXIT(
        "StridedSlice: Expected begin, end, and strides to be 1D equal size "
        "tensors, which size is equal to input_tensor dimensionality"
        "Input dimensionality:\"%u\", begin_tensor size: \"%u\", end_tensor "
        "size: \"%u\" and strides_tensor size: \"%u\"\n",
        input.get_shape().num_dims(), begin_tensor_size, end_tensor_size,
        strides_tensor_size);
  }

  if (ellipsis_mask) {
    ERR_EXIT(
        "StridedSlice: ellipsis mask should be handled by uTensor code "
        "generator and hence not supported in the runtime\n");
  }

  if (new_axis_mask) {
    ERR_EXIT(
        "StridedSlice: new axis mask should be handled by uTensor code "
        "generator and hence not supported in the runtime\n")
  }
  for (uint32_t i = 0; i < begin_tensor_size; ++i) {
    if (shrink_axis_mask & (1 << i)) {
      int32_t begin_ = begin_tensor(i);
      int32_t end_ = end_tensor(i);
      int32_t stride = strides_tensor(i);
      int32_t dim_size = (end_ - begin_) / stride;
      if (dim_size != 1) {
        ERR_EXIT(
            "StridedSlice: invalid shrink mask (%i). %uth output shape should "
            "be 1, "
            "get %i\n",
            shrink_axis_mask, i, dim_size);
      }
    }
  }
}
}  // namespace ReferenceOperators
}  // namespace uTensor