#include "ReduceFunc.hpp"

#include <algorithm>

namespace uTensor {
namespace ReferenceOperators {
ReduceOperator::ReduceOperator(initializer_list<uint16_t> dims) {
  size_t i = 0;
  for (auto s : dims) {
    _dims[i++] = s;
  }
}
uint32_t ReduceOperator::adjust_linear_idx(Tensor& tensor, uint32_t idx) {
  TensorShape ori_shape = tensor->get_shape();
  TensorStrides ori_strides = TensorStrides(ori_shape);
  TensorShape reduced_shape = TensorShape(0, 0, 0, 0);
  int c = 0;
  for (int i = 0; i < 4; ++i) {
    bool is_reduce_dim = false;
    for (size_t j = 0; j < 4; ++j) {
      if (i == _dims[j]) {
        is_reduce_dim = true;
        break;
      }
    }
    if (!is_reduce_dim) {
      reduced_shape[c] = ori_shape[i];
      c++;
    }
  }
  reduced_shape.update_dims();
  TensorStrides reduced_strides = TensorStrides(reduced_shape);
  uint32_t new_idx = 0;
  size_t current_idx = 0;
  uint32_t residual = idx;
  for (size_t i = 0; i < ori_shape.num_dims(); ++i) {
    uint32_t axis_size = ori_shape[i];
    uint32_t stride = ori_strides[i];
    uint32_t q = std::min(residual / stride, axis_size - 1);
    bool is_reduce_dim = false;
    for (auto d : _dims) {
      if (d == i) {
        is_reduce_dim = true;
        break;
      }
    }
    if (!is_reduce_dim) {
      new_idx += reduced_strides[current_idx] * q;
      current_idx++;
    }
    residual -= q * stride;
  }
  return new_idx;
}

template <>
void ReduceMeanOperator<int8_t>::compute() {
  Tensor& inputT = inputs[input].tensor();
  Tensor& outputT = outputs[output].tensor();
  for (uint32_t i = 0; i < outputT->num_elems(); ++i) {
    outputT(i) = static_cast<int8_t>(0);
  }
  float denum = 1;
  for (auto d : _dims) {
    denum *= inputT->get_shape()[d];
  }
  const float iscale = inputT->get_quantization_params().get_scale_for_channel(0);
  const int32_t izp = inputT->get_quantization_params().get_zeroP_for_channel(0);
  const float oscale = outputT->get_quantization_params().get_scale_for_channel(0);
  const int32_t ozp = outputT->get_quantization_params().get_zeroP_for_channel(0);
  for (uint32_t offset = 0; offset < inputT->num_elems(); ++offset) {
    uint32_t new_offset = adjust_linear_idx(input, offset);
    //outputT(new_offset) += inputT(offset) / denum
    const int32_t iv8 = static_cast<int8_t>(inputT(offset));
    const int32_t ov8 = static_cast<int8_t>(outputT(new_offset));
    const float input_value = (iv8 - izp)*iscale / denum;
    const float out_val = (ov8 - ozp)*oscale + input_value;
    
    const int32_t otmp = static_cast<int32_t>(out_val/oscale) + ozp;
    const int8_t out8 = (otmp < -127 ) ? -128 : (otmp > 127) ? 127 : static_cast<int8_t>(otmp);
    outputT(new_offset) = out8;
  }
}


}  // namespace ReferenceOperators
}  // namespace uTensor
