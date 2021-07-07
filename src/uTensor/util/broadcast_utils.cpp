#include "uTensor/util/broadcast_utils.hpp"

#include "uTensor/core/uTensor_util.hpp"

namespace uTensor {

Broadcaster::Broadcaster(const TensorShape& shape1, const TensorShape& shape2)
    : Broadcaster(const_cast<TensorShape&>(shape1),
                  const_cast<TensorShape&>(shape2)) {}

Broadcaster::Broadcaster(TensorShape& shape1, TensorShape& shape2)
    : _shape1(shape1),
      _shape2(shape2),
      _hit_last(false),
      _promoted_num_dims(0) {
  _reset();
  for (size_t i = 0; i < UTENSOR_MAX_NDIMS; ++i) {
    _promoted_shape[i] = 0;
  }
  if (broadcastable(_shape1, _shape2)) {
    uint8_t ndims1 = _shape1.num_dims();
    uint8_t ndims2 = _shape2.num_dims();
    _promoted_num_dims = ndims1 > ndims2 ? ndims1 : ndims2;
    for (size_t i = 1; i <= _promoted_num_dims; ++i) {
      int idx1 = ndims1 - i;
      int idx2 = ndims2 - i;
      uint16_t size1 = idx1 >= 0 ? _shape1[idx1] : 1;
      uint16_t size2 = idx2 >= 0 ? _shape2[idx2] : 1;
      _promoted_shape[_promoted_num_dims - i] = size1 > size2 ? size1 : size2;
    }
    {
      TensorStrides tensor_strides(_shape1);
      for (size_t i = 1; i <= _promoted_num_dims; ++i) {
        int idx = ndims1 - i;
        uint32_t s = 0;
        if (idx >= 0 &&
            shape1[idx] == _promoted_shape[_promoted_num_dims - i]) {
          s = tensor_strides[idx];
        }
        _strides1[_promoted_num_dims - i] = s;
      }
    }
    {
      TensorStrides tensor_strides(_shape2);
      for (size_t i = 1; i <= _promoted_num_dims; ++i) {
        int idx = ndims2 - i;
        uint32_t s = 0;
        if (idx >= 0 &&
            shape2[idx] == _promoted_shape[_promoted_num_dims - i]) {
          s = tensor_strides[idx];
        }
        _strides2[_promoted_num_dims - i] = s;
      }
    }
  } else {
    uTensor_printf("[WARN] unbroadcastable shapes given:\n");
    shape1.print(false);
    shape2.print(true);
  }
}

TensorShape Broadcaster::promoted_shape() const {
  return TensorShape(_promoted_shape, _promoted_num_dims);
}

void Broadcaster::next(int32_t& linear_index1, int32_t& linear_index2) {
  if (_promoted_num_dims == 0) {
    // not broadcastable case
    linear_index1 = -1;
    linear_index2 = -1;
    return;
  }
  // compute linear indices at the moment
  int32_t acc1 = 0, acc2 = 0;
  for (size_t i = 0; i < _promoted_num_dims; ++i) {
    acc1 += _idx_cnt[i] * _strides1[i];
    acc2 += _idx_cnt[i] * _strides2[i];
  }
  linear_index1 = acc1;
  linear_index2 = acc2;
  // update indices
  _idx_cnt[_promoted_num_dims - 1] += 1;
  for (int i = _promoted_num_dims - 2; i >= 0; --i) {
    if (_idx_cnt[i + 1] >= _promoted_shape[i + 1]) {
      _idx_cnt[i] += 1;
    }
  }
  // check if it's done
  bool done = _is_done();
  if (_hit_last || done) {
    if (_hit_last) {
      _reset();
      linear_index1 = -1;
      linear_index2 = -1;
      return;
    } else {
      _hit_last = true;
    }
  }
  // check if the idx exceed end
  for (size_t i = 0; i < _promoted_num_dims; ++i) {
    if (_idx_cnt[i] >= _promoted_shape[i]) {
      _idx_cnt[i] = 0;
    }
  }
}

bool Broadcaster::broadcastable(const TensorShape& shape1,
                                const TensorShape& shape2) {
  if (shape1 == shape2) {
    return true;
  }
  bool is_ok = true;
  uint8_t ndims1 = shape1.num_dims();
  uint8_t ndims2 = shape2.num_dims();
  uint8_t max_ndims = ndims1 > ndims2 ? ndims1 : ndims2;
  for (int i = 1; i <= max_ndims; ++i) {
    int idx1 = ndims1 - i;
    int idx2 = ndims2 - i;
    uint16_t size1 = idx1 >= 0 ? shape1[idx1] : 1;
    uint16_t size2 = idx2 >= 0 ? shape2[idx2] : 1;
    if (size1 != size2) {
      if (size1 != 1 and size2 != 1) {
        is_ok = false;
        break;
      }
    }
  }
  return is_ok;
}

void Broadcaster::_reset() {
  for (size_t i = 0; i < UTENSOR_MAX_NDIMS; ++i) {
    _idx_cnt[i] = 0;
  }
  _hit_last = false;
}

bool Broadcaster::_is_done() {
  bool done = true;
  size_t i = 0;
  for (size_t i = 0; i < _promoted_num_dims; ++i) {
    if (_idx_cnt[i] < _promoted_shape[i]) {
      done = false;
      break;
    }
  }
  return done;
}

}  // namespace uTensor