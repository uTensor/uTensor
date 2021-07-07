#ifndef UTENSOR_UTIL_BROADCAST_H
#define UTENSOR_UTIL_BROADCAST_H

#include "uTensor/core/tensor.hpp"

namespace uTensor {

class Broadcaster {
 public:
  Broadcaster(TensorShape& shape1, TensorShape& shape2);
  Broadcaster(const TensorShape& shape1, const TensorShape& shape2);

  static bool broadcastable(const TensorShape& shape1,
                            const TensorShape& shape2);
  void next(int32_t& linear_index1, int32_t& linear_index2);
  TensorShape promoted_shape() const;

 private:
  void _reset();
  bool _is_done();

 private:
  int32_t _idx_cnt[UTENSOR_MAX_NDIMS];
  bool _hit_last;
  uint16_t _promoted_shape[UTENSOR_MAX_NDIMS];
  uint8_t _promoted_num_dims;
  uint32_t _strides1[UTENSOR_MAX_NDIMS];
  uint32_t _strides2[UTENSOR_MAX_NDIMS];
  TensorShape _shape1;
  TensorShape _shape2;
};
}  // namespace uTensor

#endif  // UTENSOR_UTIL_BROADCAST_H
