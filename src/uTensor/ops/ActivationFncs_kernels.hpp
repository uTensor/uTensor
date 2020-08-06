#ifndef UTENSOR_ACTIVATIONS_KERNELS_H
#define UTENSOR_ACTIVATIONS_KERNELS_H
#include "uTensor/core/operatorBase.hpp"
#include <cmath>
#include <limits>
#include <functional>

using std::exp;

namespace uTensor {

namespace Fuseable {

  template <typename T>
  using Activation = std::function<T(T)>;
  
  template <typename T>
  T NoActivation(T x) { return x; }
  
  template <typename T>
  T ReLU(T x) { return (x < 0) ? 0 : x; }
  
  template <typename T>
  T ReLU6(T x) { 
    if (x < 0){
      return 0;
    } else if (x > 6) {
      return 6;
    } else {
      return x;
    }
  }
  
  template <typename T>
  T Sigmoid(T x) {
    const T one = 1;
    return one / ( one + exp(-x) );
  }

} // namespace Fuseable

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

template <typename T>
void inplace_relu6_k(Tensor& t) {
  T tmp;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    tmp = t(i);
    if (tmp < 0) {
      t(i) = static_cast<T>(0);
    }
    if (tmp > 6) {
      t(i) = static_cast<T>(6);
    }
  }
}

template <typename T>
void relu6_k(Tensor& out, const Tensor& in) {
  T tmp;
  uint32_t in_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < in_size; i++) {
    tmp = in(i);
    if (tmp < 0) {
      tmp = static_cast<T>(0);
    }
    if (tmp > 6) {
      tmp = static_cast<T>(6);
    }
    out(i) = tmp;
  }
}

}  // namespace uTensor
#endif
