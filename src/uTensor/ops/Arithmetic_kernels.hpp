#ifndef UTENSOR_ARITH_KERNELS_H
#define UTENSOR_ARITH_KERNELS_H
#include "uTensor/core/operatorBase.hpp"

namespace uTensor {
template <typename T>
void add_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // Decide on c shape
  TensorShape c_shape = c->get_shape();
  uint32_t c_size = c_shape.get_linear_size();
  // TensorInterface& C = reinterpret_cast<TensorInterface*>(*c);
  // const TensorInterface& A = reinterpret_cast<TensorInterface*>(*a);
  // const TensorInterface& B = reinterpret_cast<TensorInterface*>(*b);

  for (uint32_t i = 0; i < c_size; i++)
    c(i) = static_cast<T>(static_cast<T>(a(i)) + static_cast<T>(b(i)));
}

template <typename T>
void sub_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // Decide on c shape
  TensorShape c_shape = c->get_shape();
  uint32_t c_size = c_shape.get_linear_size();
  // TensorInterface& C = reinterpret_cast<TensorInterface*>(*c);
  // const TensorInterface& A = reinterpret_cast<TensorInterface*>(*a);
  // const TensorInterface& B = reinterpret_cast<TensorInterface*>(*b);

  for (uint32_t i = 0; i < c_size; i++)
    c(i) = static_cast<T>(static_cast<T>(a(i)) - static_cast<T>(b(i)));
}

template <typename T>
void mul_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // Decide on c shape
  TensorShape c_shape = c->get_shape();
  uint32_t c_size = c_shape.get_linear_size();
  // TensorInterface& C = reinterpret_cast<TensorInterface*>(*c);
  // const TensorInterface& A = reinterpret_cast<TensorInterface*>(*a);
  // const TensorInterface& B = reinterpret_cast<TensorInterface*>(*b);

  for (uint32_t i = 0; i < c_size; i++)
    c(i) = static_cast<T>(static_cast<T>(a(i)) * static_cast<T>(b(i)));
}

}  // namespace uTensor
#endif
