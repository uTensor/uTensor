#ifndef UTENSOR_ACTIVATIONS_KERNELS_H
#define UTENSOR_ACTIVATIONS_KERNELS_H
#include "operatorBase.hpp"
#include <cmath>
#include <limits>

using std::exp;

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

template <typename T>
void inplace_softmax_k(Tensor& in, T beta = 1) {
  T tmp;
  T mSum = 0;
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(i, j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(i,j)) - max) * beta);
      mSum += tmp;
      in(i,j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      in(i, j)  = static_cast<T>(in(i, j)) / mSum;
    }
  }
}
template <typename T>
void softmax_k(Tensor& out, const Tensor& in, T beta=1) {
  T tmp;
  T mSum = 0;
  const TensorShape& inShape = in->get_shape();
  int outer_dim = inShape.num_dims() -1;
  int depth = inShape[outer_dim];
  int out_side_numelems = 1;
  for(int i = 0; i < inShape.num_dims(); i++){
    out_side_numelems *= (i == outer_dim) ? 1: inShape[i];
  }

  for (int i = 0; i < out_side_numelems; i++) {
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    T max = std::numeric_limits<T>::lowest();
    for(int j = 0; j < depth; j++){
      max = std::max(max, static_cast<T>(in(i, j)));
    }

    T mSum = 0;
    for(int j = 0; j < depth; j++){
      T tmp = exp((static_cast<T>(in(i,j)) - max) * beta);
      mSum += tmp;
      out(i,j) = tmp;
    }
    for(int j = 0; j < depth; j++){
      out(i, j)  = static_cast<T>(out(i, j)) / mSum;
    }
  }

}
template <typename T>
void inplace_sigmoid_k(Tensor& t) {
  const T one = 1;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(t(i))));
    t(i) = tmp;
  }
}

template <typename T>
void sigmoid_k(Tensor& out, const Tensor& in) {
  const T one = 1;
  uint32_t t_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(in(i))));
    out(i) = tmp;
  }
}

}  // namespace uTensor
#endif
