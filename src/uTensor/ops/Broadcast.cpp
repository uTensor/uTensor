#include "Broadcast.hpp"

#include <stdexcept>

Broadcaster::Broadcaster() : _output_shape(0) {}

void Broadcaster::set_shape(const TensorShape& shape_a,
                            const TensorShape& shape_b) {
  if (!is_broadcastable(shape_a, shape_b, _output_shape)) {
    throw std::runtime_error("Shapes are not broadcastable");
  }
  set_shape(shape_a, shape_b, _output_shape);
}

/*
! Use with caution
This function is assuming that the shapes are broadcastable and the output
shape is already set correctly ! In case that the output shape is not known, use
`set_shape(const TensorShape& shape_a, const TensorShape& shape_b)` instead
*/
void Broadcaster::set_shape(const TensorShape& shape_a,
                            const TensorShape& shape_b,
                            const TensorShape& output_shape) {
  uint8_t num_dims_a = shape_a.num_dims(), num_dims_b = shape_b.num_dims(),
          num_dims_c = output_shape.num_dims();
  for (uint8_t i = 0; i < num_dims_c; i++) {
    _output_shape[i] = output_shape[i];
  }
  _output_shape.update_dims();
  ouput_strides = TensorStrides(_output_shape);
  uint32_t s_a = 1, s_b = 1;
  for (uint8_t offset = 0; offset < num_dims_c; offset++) {
    int c_idx = num_dims_c - offset - 1, a_idx = num_dims_a - offset - 1,
        b_idx = num_dims_b - offset - 1;
    if (a_idx < 0) {
      _strides_a[c_idx] = 0;
    } else if (shape_a[a_idx] == 1) {
      _strides_a[c_idx] = 0;
    } else {
      _strides_a[c_idx] = s_a;
      s_a *= shape_a[a_idx];
    }
    if (b_idx < 0) {
      _strides_b[c_idx] = 0;
    } else if (shape_b[b_idx] == 1) {
      _strides_b[c_idx] = 0;
    } else {
      _strides_b[c_idx] = s_b;
      s_b *= shape_b[b_idx];
    }
  }
}

std::pair<uint32_t, uint32_t> Broadcaster::get_linear_idx(
    uint32_t idx_c) const {
  if (idx_c >= _output_shape.get_linear_size()) {
    throw std::runtime_error("Index out of bounds");
  }
  uint32_t idx_a = 0, idx_b = 0;
  for (uint8_t i = 0; i < _output_shape.num_dims(); i++) {
    idx_a += (idx_c / ouput_strides[i]) * _strides_a[i];
    idx_b += (idx_c / ouput_strides[i]) * _strides_b[i];
    idx_c %= ouput_strides[i];
  }
  return std::make_pair(idx_a, idx_b);
}

bool is_broadcastable(const TensorShape& shape_a, const TensorShape& shape_b,
                      TensorShape& output_shape) {
  uint8_t num_dims_a = shape_a.num_dims(), num_dims_b = shape_b.num_dims();
  uint8_t num_dims_c = std::max(num_dims_a, num_dims_b);

  // Check if the shapes are compatible
  // Each tensor has at least one dimension.
  // When iterating over the dimension sizes, starting at the trailing
  // dimension, the dimension sizes must either be equal, one of them is 1, or
  // one of them does not exist.
  for (uint8_t offset = 0; offset < num_dims_c; offset++) {
    int c_idx = num_dims_c - offset - 1, a_idx = num_dims_a - offset - 1,
        b_idx = num_dims_b - offset - 1;
    uint16_t size_a = (a_idx < 0) ? 1 : shape_a[a_idx];
    uint16_t size_b = (b_idx < 0) ? 1 : shape_b[b_idx];
    if (size_a != size_b && size_a != 1 && size_b != 1) {
      return false;
    }
    output_shape[c_idx] = std::max(size_a, size_b);
  }
  output_shape.update_dims();
  return true;
}