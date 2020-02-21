#ifndef UTENSOR_ACTIVATIONS_OPS_H
#define UTENSOR_ACTIVATIONS_OPS_H
#include "operatorBase.hpp"
#include <type_traits>

namespace uTensor {
class InPlaceActivationFnc : public OperatorInterface<1,0> {

  public:
    enum names_in: uint8_t { x };
  protected:
    virtual void compute() = 0;
};

template <typename T>
class InPlaceReLU : public InPlaceActivationFnc {
  // ReLU only makes sense if there is a notion of negative
  static_assert(std::is_signed<T>::value, "Error attempted to construct ReLU on non-signed types");

  protected:
    virtual void compute() {
      Tensor& t = *inputs[x].tensor;
      uint32_t t_size = t->get_shape().get_linear_size();
      T tmp;
      for (uint32_t i = 0; i < t_size; i++){
        tmp = t(i);
        if(tmp < 0){
          t(i) = static_cast<T>(0);
        }
      }
    }
};




}

#endif
