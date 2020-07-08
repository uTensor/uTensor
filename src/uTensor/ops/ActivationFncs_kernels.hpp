#ifndef UTENSOR_ACTIVATIONS_KERNELS_H
#define UTENSOR_ACTIVATIONS_KERNELS_H
#include "operatorBase.hpp"
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
class inplace_relu_k_impl {
  public:
    inplace_relu_k_impl() {}
    void operator()(Tensor& t) const;
};

template <typename T>
void inplace_relu_k_impl<T>::operator()(Tensor& t) const {
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
class relu_k_impl{
  public:
    void operator()(Tensor& out, const Tensor& in) const;
};

template <typename T>
void relu_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
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
class inplace_relu6_k_impl {
  public:
    void operator()(Tensor& t) const;

};

template <typename T>
void inplace_relu6_k_impl<T>::operator()(Tensor& t) const {
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
class relu6_k_impl {
  public:
    void operator()(Tensor& out, const Tensor& in) const;
};

template <typename T>
void relu6_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
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

void sq_softmax_k(Tensor& out, const Tensor& in, int8_t beta=1);

template <typename T>
class inplace_sigmoid_k {
  public:
    void operator()(Tensor& t) const;
};

template <typename T>
void inplace_sigmoid_k<T>::operator()(Tensor& t) const {
  const T one = 1;
  uint32_t t_size = t->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(t(i))));
    t(i) = tmp;
  }
}

template <typename T>
class sigmoid_k_impl {
  public:
    void operator()(Tensor& out, const Tensor& in) const;

};

template <typename T>
void sigmoid_k_impl<T>::operator()(Tensor& out, const Tensor& in) const {
  const T one = 1;
  uint32_t t_size = in->get_shape().get_linear_size();
  for (uint32_t i = 0; i < t_size; i++) {
    const T tmp = one / (one + exp(- static_cast<T>(in(i))));
    out(i) = tmp;
  }
}

template <>
void sigmoid_k_impl<int8_t>::operator()(Tensor& out, const Tensor& in) const;

// Set defaults
template <typename T>
using sigmoid_k = sigmoid_k_impl<T>();

template <typename T>
using inplace_relu_k = inplace_relu_k_impl<T>;
template <typename T>
using relu_k = relu_k_impl<T>();
template <typename T>
using inplace_relu6_k = inplace_relu6_k_impl<T>();
template <typename T>
using relu6_k = relu6_k_impl<T>();


}  // namespace uTensor
#endif
