#ifndef UTENSOR_CMSIS_ACTIVATION_FUNCTIONS
#define UTENSOR_CMSIS_ACTIVATION_FUNCTIONS
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


template<typename T>
void cmsis_relu_select(T* data, const uint16_t numel);
template<>
void cmsis_relu_select<q7_t>(q7_t* data, const uint16_t numel){
    arm_relu_q7(data, numel);
}
template<>
void cmsis_relu_select<q15_t>(q15_t* data, const uint16_t numel){
    arm_relu_q15(data, numel);
}

/**
 * @param [in] data input tensor
 */
template<typename T>
void ReluCmsis(S_TENSOR data, S_TENSOR out, T x)
{
    out = data;
    T* d = out->write<T>(0, sizeof(T));
    const uint16_t numel = out->getShape()[0];
    cmsis_relu_select(d, numel);
}

template <class T>
class ReluCmsisOp : public Operator {
  public:
  ReluCmsisOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
      T x;
      ReluCmsis(inputs[0], outputs[0], x);
  }
};


#endif 
