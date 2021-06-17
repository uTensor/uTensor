#ifndef UTENSOR_ARITH_KERNELS_H
#define UTENSOR_ARITH_KERNELS_H
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/util/broadcast_utils.hpp"

namespace uTensor {
template <typename T>
void add_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // check if need broadcasting
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  if (a_shape == b_shape) {
    // Decide on c shape
    TensorShape& c_shape = c->get_shape();
    for (uint32_t i = 0; i < c_shape.num_elems(); i++)
      c(i) = static_cast<T>(static_cast<T>(a(i)) + static_cast<T>(b(i)));
  } else {
    if (!Broadcaster::broadcastable(a_shape, b_shape)) {
      uTensor_printf("unprocastable inputs for elementwise add\n");
      Context::get_default_context()->throwError(new UnbroadcastableShapeError);
      return;
    }
    Broadcaster broad(a_shape, b_shape);
    int32_t linear_idx_a = 0, linear_idx_b = 0;
    for (uint32_t i = 0; i < broad.promoted_shape().num_elems(); ++i) {
      broad.next(linear_idx_a, linear_idx_b);
      T value_a = a(linear_idx_a), value_b = b(linear_idx_b);
      c(i) = value_a + value_b;
    }
  }
}

template <typename T>
void sub_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // check if need broadcasting
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  if (a_shape == b_shape) {
    // Decide on c shape
    TensorShape c_shape = c->get_shape();
    for (uint32_t i = 0; i < c_shape.num_elems(); i++)
      c(i) = static_cast<T>(static_cast<T>(a(i)) - static_cast<T>(b(i)));
  } else {
    if (!Broadcaster::broadcastable(a_shape, b_shape)) {
      uTensor_printf("unprocastable inputs for elementwise add\n");
      Context::get_default_context()->throwError(new UnbroadcastableShapeError);
      return;
    }
    Broadcaster broad(a_shape, b_shape);
    int32_t linear_idx_a = 0, linear_idx_b = 0;
    for (uint32_t i = 0; i < broad.promoted_shape().num_elems(); ++i) {
      broad.next(linear_idx_a, linear_idx_b);
      T value_a = a(linear_idx_a), value_b = b(linear_idx_b);
      c(i) = value_a - value_b;
    }
  }
}

template <typename T>
void mul_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // check if need broadcasting
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  if (a_shape == b_shape) {
    // Decide on c shape
    TensorShape c_shape = c->get_shape();
    for (uint32_t i = 0; i < c_shape.num_elems(); i++)
      c(i) = static_cast<T>(static_cast<T>(a(i)) * static_cast<T>(b(i)));
  } else {
    if (!Broadcaster::broadcastable(a_shape, b_shape)) {
      uTensor_printf("unprocastable inputs for elementwise add\n");
      Context::get_default_context()->throwError(new UnbroadcastableShapeError);
      return;
    }
    Broadcaster broad(a_shape, b_shape);
    int32_t linear_idx_a = 0, linear_idx_b = 0;
    for (uint32_t i = 0; i < broad.promoted_shape().num_elems(); ++i) {
      broad.next(linear_idx_a, linear_idx_b);
      T value_a = a(linear_idx_a), value_b = b(linear_idx_b);
      c(i) = value_a * value_b;
    }
  }
}

template <typename T>
void div_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // check if need broadcasting
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  if (a_shape == b_shape) {
    // Decide on c shape
    TensorShape c_shape = c->get_shape();
    for (uint32_t i = 0; i < c_shape.num_elems(); i++)
      c(i) = static_cast<T>(static_cast<T>(a(i)) / static_cast<T>(b(i)));
  } else {
    if (!Broadcaster::broadcastable(a_shape, b_shape)) {
      uTensor_printf("unprocastable inputs for elementwise add\n");
      Context::get_default_context()->throwError(new UnbroadcastableShapeError);
      return;
    }
    Broadcaster broad(a_shape, b_shape);
    int32_t linear_idx_a = 0, linear_idx_b = 0;
    for (uint32_t i = 0; i < broad.promoted_shape().num_elems(); ++i) {
      broad.next(linear_idx_a, linear_idx_b);
      T value_a = a(linear_idx_a), value_b = b(linear_idx_b);
      c(i) = value_a / value_b;
    }
  }
}

}  // namespace uTensor
#endif
