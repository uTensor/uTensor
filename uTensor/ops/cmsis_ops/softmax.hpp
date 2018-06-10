#ifndef UTENSOR_CMSIS_SOFTMAX_FUNCTIONS
#define UTENSOR_CMSIS_SOFTMAX_FUNCTIONS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


/**
 * @param [in] data input tensor
 */
template<typename T1, typename TOut>
void SoftmaxCmsis(S_TENSOR data, S_TENSOR out)
{
    //Throw error if this gets called
}

template<>
void SoftmaxCmsis<q7_t, q7_t>(S_TENSOR data, S_TENSOR out)
{
    const q7_t* in = data->read<q7_t>(0, sizeof(q7_t));
    out->resize(in->getShape());
    q7_t* d = out->write<q7_t>(0, sizeof(q7_t));
    uint16_t numel = (uint16_t)in->getSize();
    arm_softmax_q7(in, numel, d);
    //Error checking
}

template<>
void SoftmaxCmsis<q15_t, q15_t>(S_TENSOR data, S_TENSOR out)
{
    const q15_t* in = data->read<q15_t>(0, sizeof(q15_t));
    out->resize(in->getShape());
    q15_t* d = out->write<q15_t>(0, sizeof(q15_t));
    uint16_t numel = (uint16_t)in->getSize();
    arm_softmax_q15(in, numel, d);
    //Error checking
}


template <class T1, class TOut>
class SoftmaxCmsisOp : public Operator {
  public:
  SoftmaxCmsisOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
      SoftmaxCmsis<T1, TOut>(inputs[0], outputs[0]);
  }
};


#endif 
