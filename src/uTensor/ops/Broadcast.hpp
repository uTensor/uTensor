#pragma once

#include <utility>

#include "uTensor/core/types.hpp"

class Broadcaster {
 public:
  Broadcaster();
  void set_shape(const TensorShape& shape_a, const TensorShape& shape_b);
  void set_shape(const TensorShape& shape_a, const TensorShape& shape_b,
                 const TensorShape& shape_c);
  std::pair<uint32_t, uint32_t> get_linear_idx(uint32_t idx_c) const;
  TensorShape get_shape_c() const { return _shape_c; }

 private:
  TensorShape _shape_c;
  TensorStrides _strides_a;
  TensorStrides _strides_b;
  TensorStrides _strides_c;
};

bool is_broadcastable(const TensorShape& shape_a, const TensorShape& shape_b,
                      TensorShape& shape_c);
