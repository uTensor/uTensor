#pragma once

#include <utility>

#include "uTensor/core/types.hpp"

class Broadcaster {
 public:
  Broadcaster();
  void set_shape(const TensorShape& shape_a, const TensorShape& shape_b);
  /*
  ! Use with caution
  This function is assuming that the shapes are broadcastable and the output
  shape is already set correctly ! In case that the output shape is not known,
  use `set_shape(const TensorShape& shape_a, const TensorShape& shape_b)`
  instead
  */
  void set_shape(const TensorShape& shape_a, const TensorShape& shape_b,
                 const TensorShape& output_shape);
  std::pair<uint32_t, uint32_t> get_linear_idx(uint32_t idx_c) const;
  TensorShape get_output_shape() const { return _output_shape; }

 private:
  TensorShape _output_shape;
  TensorStrides _strides_a;
  TensorStrides _strides_b;
  TensorStrides ouput_strides;
};

bool is_broadcastable(const TensorShape& shape_a, const TensorShape& shape_b,
                      TensorShape& output_shape);
