#ifndef UTENSOR_CMSIS_SOFTMAX_FUNCTIONS
#define UTENSOR_CMSIS_SOFTMAX_FUNCTIONS
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


template<typename T1>
void cmsis_softmax_selector(const T1* vec_in, const uint16_t dim_vec, T1* p_out);
template<>
void cmsis_softmax_selector<q15_t>(const q15_t* vec_in, const uint16_t dim_vec, q15_t* p_out)
{
    arm_softmax_q15(vec_in, dim_vec, p_out);
}
template<>
void cmsis_softmax_selector<q7_t>(const q7_t* vec_in, const uint16_t dim_vec, q7_t* p_out)
{
    arm_softmax_q7(vec_in, dim_vec, p_out);
}

/**
 * @param [in] data input tensor
 */
template<typename T1>
void SoftmaxCmsis(S_TENSOR data, S_TENSOR out, T1 x){
    const T1* in = data->read<T1>(0, sizeof(T1));
    const uint16_t numel = data->getShape()[0];
    out->resize(data->getShape());
    T1* d = out->write<T1>(0, sizeof(T1));
    cmsis_softmax_selector(in, numel, d);

}

template <class T1>
class SoftmaxCmsisOp : public Operator {
  public:
  SoftmaxCmsisOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
      T1 x;
      SoftmaxCmsis(inputs[0], outputs[0], x);
  }
};


#endif 
