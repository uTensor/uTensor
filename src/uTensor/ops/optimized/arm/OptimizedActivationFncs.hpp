#ifndef CMSIS_UTENSOR_ACTIVATIONS_OPS_H
#define CMSIS_UTENSOR_ACTIVATIONS_OPS_H
#include <type_traits>

#include "ActivationFncs.hpp"
#include "operatorBase.hpp"
#include "arm_nnfunctions.h"

namespace uTensor {
namespace CMSIS {

template <typename T>
class InPlaceReLU : public InPlaceActivationFnc, FastOperator;

template <>
class InPlaceReLU<int8_t> : public InPlaceActivationFnc, FastOperator {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { 
    Tensor& tx = inputs[x].tensor();
    const TensorShape& shape = tx->get_shape();
    q7* data;
    uint16_t block_size = tx->get_writeable_block(data, shape.get_linear_size(), 0);
    arm_relu_q7(data, block_size);
  }
};

template <>
class InPlaceReLU<int16_t> : public InPlaceActivationFnc, FastOperator {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { 
    Tensor& tx = inputs[x].tensor();
    const TensorShape& shape = tx->get_shape();
    q16* data;
    uint16_t block_size = tx->get_writeable_block(data, shape.get_linear_size(), 0);
    arm_relu_q16(data, block_size);
  }
};

template <typename T>
class InPlaceReLU6 : public InPlaceActivationFnc, FastOperator;

template <>
class InPlaceReLU6<int8_t> : public InPlaceActivationFnc, FastOperator {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value,
                "Error attempted to construct ReLU on non-signed types");

 protected:
  virtual void compute() { 
    Tensor& tx = inputs[x].tensor();
    const TensorShape& shape = tx->get_shape();
    q7* data;
    uint16_t block_size = tx->get_writeable_block(data, shape.get_linear_size(), 0);
    arm_relu6_s8(data, block_size);
  }
};

} // CMSIS
} // uTensor
#endif
