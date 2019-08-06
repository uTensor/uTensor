#ifndef UTENSOR_CMSIS_POOLING_FUNCTIONS
#define UTENSOR_CMSIS_POOLING_FUNCTIONS
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


void cmsis_maxpool(S_TENSOR im_in_s, S_TENSOR dim_kernel_s, S_TENSOR padding_s, S_TENSOR stride,
        S_TENSOR bufferA_s, S_TENSOR im_out_s)

        q7_t *Im_in = im_in_s->write(0, sizeof(q7_t));
        const uint16_t dim_im_in = im_in_s->getShape()[0];
        const uint16_t ch_im_in = im_in_s->getShape()[2];
        *(bShift->read<uint16_t>(0,0));
        const uint16_t dim_kernel = *(dim_kernel_s->read<uint16_t>(0, 0));
        const uint16_t padding = *(padding_s->read<uint16_t>(0, 0))
        const uint16_t stride = *(padding_s->read<uint16_t>(0, 0))
        const uint16_t dim_im_out = im_out_s->getShape()[0];
        q7_t *bufferA = bufferA_s->write(0, sizeof(q7_t));
        q7_t *Im_out  = im_out_s->write(0, sizeof(q7_t));

    arm_maxpool_q7_HWC(
        Im_in, 
        dim_im_in,
        ch_im_in,
        dim_kernel,
        padding,
        stride,
        dim_im_out,
        bufferA,
        Im_out)
}

void cmsis_avepool(S_TENSOR im_in_s, S_TENSOR dim_kernel_s, S_TENSOR padding_s, S_TENSOR stride,
        S_TENSOR bufferA_s, S_TENSOR im_out_s)

        q7_t *Im_in = im_in_s->write(0, sizeof(q7_t));
        const uint16_t dim_im_in = im_in_s->getShape()[0];
        const uint16_t ch_im_in = im_in_s->getShape()[2];
        *(bShift->read<uint16_t>(0,0));
        const uint16_t dim_kernel = *(dim_kernel_s->read<uint16_t>(0, 0));
        const uint16_t padding = *(padding_s->read<uint16_t>(0, 0))
        const uint16_t stride = *(padding_s->read<uint16_t>(0, 0))
        const uint16_t dim_im_out = im_out_s->getShape()[0];
        q7_t *bufferA = bufferA_s->write(0, sizeof(q7_t));
        q7_t *Im_out  = im_out_s->write(0, sizeof(q7_t));

    arm_avepool_q7_HWC(
        Im_in, 
        dim_im_in,
        ch_im_in,
        dim_kernel,
        padding,
        stride,
        dim_im_out,
        bufferA,
        Im_out)
}

template <class T=q7_t>
class MaxPoolCmsisOp : public Operator {
  public:
  MaxPoolCmsisOp() {
    n_inputs = 4;
    n_outputs = 2;
  }
  virtual void compute() override {
      cmsis_maxpool(inputs[0], 
              inputs[1],
              inputs[2],
              inputs[3],
              inputs[4],
              outputs[0], outputs[1]);
  }
};

template <class T=q7_t>
class AvePoolCmsisOp : public Operator {
  public:
  AvePoolCmsisOp() {
    n_inputs = 4;
    n_outputs = 2;
  }
  virtual void compute() override {
      cmsis_avepool(inputs[0], 
              inputs[1],
              inputs[2],
              inputs[3],
              inputs[4],
              outputs[0], outputs[1]);
  }
};


#endif 
