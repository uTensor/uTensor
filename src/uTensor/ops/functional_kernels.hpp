#ifndef UTENSOR_FUNCTION_KERNELS_H
#define UTENSOR_FUNCTION_KERNELS_H
#include <algorithm>
#include <limits>
#include <vector>

#include "operatorBase.hpp"
namespace uTensor {

template <typename T>
T min_kernel(const Tensor& a) {
  T tmp = std::numeric_limits<T>::max();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for (uint32_t i = 0; i < a_size; i++) {
    tmp = std::min(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}
template <typename T>
void min_kernel(Tensor& out, const Tensor& a) {
  out(0) = min_kernel<T>(a);
}

template <typename T>
T max_kernel(const Tensor& a) {
  T tmp = std::numeric_limits<T>::lowest();
  const TensorShape& a_shape = a->get_shape();
  const uint32_t a_size = a_shape.get_linear_size();
  for (uint32_t i = 0; i < a_size; i++) {
    tmp = std::max(tmp, static_cast<T>(a(i)));
  }
  return tmp;
}

template <typename T>
void max_kernel(Tensor& out, const Tensor& a) {
  out(0) = max_kernel<T>(a);
}

template <typename T>
void squeeze_kernel(Tensor& in, std::vector<uint8_t> axis) {
  T dims[4];
  memset(&dims, 0, 4*sizeof(T));
  TensorShape& shape = in->get_shape();
  // TODO optimize
  int dim_cursor = 0;
  for(int i = 0; i < 4; i++){
    if(shape[i] == 1) {
      if(axis.size() > 0){
        // Decide if we should keep or move on
        std::vector<uint8_t>::iterator it = std::find(axis.begin(), axis.end(), i);
        if(it == axis.end()){
          dims[dim_cursor] = shape[i];
          dim_cursor++;
        }
      }
    } else {
      dims[dim_cursor] = shape[i];
      dim_cursor++;
    }
  }
  for(int i = 0; i < 4; i++){
    shape[i] = dims[i];
  }
  
}

}  // namespace uTensor
#endif
