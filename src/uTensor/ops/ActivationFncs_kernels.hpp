#ifndef UTENSOR_ACTIVATIONS_KERNELS_H
#define UTENSOR_ACTIVATIONS_KERNELS_H
#include "operatorBase.hpp"

namespace uTensor {

template <typename T>
void inplace_relu_k(Tensor& t) {
  T tmp;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    tmp = t(i);
    if (tmp < 0) {
      t(i) = static_cast<T>(0);
    }
  }
}

template <typename T>
void relu_k(Tensor& out, const Tensor& in) {
  T tmp;
  uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    tmp = in(i);
    if (tmp < 0) {
      tmp = static_cast<T>(0);
    }
    out(i) = tmp;
  }
}

}  // namespace uTensor
#endif
