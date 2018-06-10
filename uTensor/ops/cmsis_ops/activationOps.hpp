#ifndef UTENSOR_CMSIS_ACTIVATION_FUNCTIONS
#define UTENSOR_CMSIS_ACTIVATION_FUNCTIONS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


/**
 * @param [in] data input tensor
 */
template<typename T1, typename TOut>
void ReluCmsis(S_TENSOR data, S_TENSOR out)
{
    //Throw error if this gets called
}

template<>
void ReluCmsis<q7_t, q7_t>(S_TENSOR data, S_TENSOR out)
{
    out = data;
    q7_t* d = out->write<q7_t>(0, sizeof(q7_t));
    uint16_t numel = (uint16_t)out->getSize();
    arm_relu_q7(d, numel);
    //Error checking
}

template<>
void ReluCmsis<q15_t, q15_t>(S_TENSOR data, S_TENSOR out)
{
    out = data;
    q15_t* d = out->write<q15_t>(0, sizeof(q15_t));
    uint16_t numel = (uint16_t)out->getSize();
    arm_relu_q15(d, numel);
    //Error checking
}

template <class T1, class TOut>
class ReluCmsisOp : public Operator {
  public:
  ReluCmsisOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
      ReluCmsis<T1, TOut>(inputs[0], outputs[0]);
  }
};


#endif 
