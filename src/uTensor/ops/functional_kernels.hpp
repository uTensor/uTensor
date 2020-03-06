#ifndef UTENSOR_FUNCTION_KERNELS_H
#define UTENSOR_FUNCTION_KERNELS_H
#include "operatorBase.hpp"
#include <algorithm>
#include <limits>
namespace uTensor {

template<typename T>
T min_kernel(const Tensor& a){
  T tmp = std::numeric_limits<T>::max();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for(uint32_t i = 0; i < a_size; i++){
    tmp = std::min(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}
template<typename T>
void min_kernel(Tensor& out, const Tensor& a){
  out(0) = min_kernel<T>(a);
}

template<typename T>
T max_kernel(const Tensor& a){
  T tmp = std::numeric_limits<T>::lowest();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for(uint32_t i = 0; i < a_size; i++){
    tmp = std::max(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}
template<typename T>
void max_kernel(Tensor& out, const Tensor& a){
  out(0) = max_kernel<T>(a);
}


}
#endif
