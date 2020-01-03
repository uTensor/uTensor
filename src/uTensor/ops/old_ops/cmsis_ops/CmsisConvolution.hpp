#ifndef UTENSOR_CMSIS_CONVOLUTION_HPP__
#define UTENSOR_CMSIS_CONVOLUTION_HPP__
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"

/***
 * Here we have two types of kernel functions. The basic function implements the function using regular GEMV approach. The opt functions operates with weights in interleaved formats.
 *
 * http://www.keil.com/pack/doc/CMSIS/NN/html/group__FC.html#gae3857bb6375692e81dde8cbd70adec08
 */

struct CmsisConvolveBasic {
    void operator()(const q15_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
            const q15_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel, 
            const uint16_t padding, const uint16_t stride, const q15_t *bias, 
            const uint16_t bias_shift, const uint16_t out_shift, q15_t *Im_out, 
            const uint16_t dim_im_out, q15_t *bufferA, q7_t *bufferB){
        arm_convolve_HWC_q15_basic(Im_in,
                dim_im_in,
                ch_im_in,
                wt,
                ch_im_out,
                dim_kernel,
                padding,
                stride,
                bias,
                bias_shift,
                out_shift,
                Im_out,
                dim_im_out,
                bufferA,
                bufferB);

    }
    void operator()(const q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
            const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel, 
            const uint16_t padding, const uint16_t stride, const q7_t *bias, 
            const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out, 
            const uint16_t dim_im_out, q15_t *bufferA, q7_t *bufferB){
        arm_convolve_HWC_q7_basic(Im_in,
                dim_im_in,
                ch_im_in,
                wt,
                ch_im_out,
                dim_kernel,
                padding,
                stride,
                bias,
                bias_shift,
                out_shift,
                Im_out,
                dim_im_out,
                bufferA,
                bufferB);

    }

};

struct CmsisConvolveFast {
    void operator()(const q15_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
            const q15_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel, 
            const uint16_t padding, const uint16_t stride, const q15_t *bias, 
            const uint16_t bias_shift, const uint16_t out_shift, q15_t *Im_out, 
            const uint16_t dim_im_out, q15_t *bufferA, q7_t *bufferB){
        arm_convolve_HWC_q15_fast(Im_in,
                dim_im_in,
                ch_im_in,
                wt,
                ch_im_out,
                dim_kernel,
                padding,
                stride,
                bias,
                bias_shift,
                out_shift,
                Im_out,
                dim_im_out,
                bufferA,
                bufferB);

    }
    void operator()(const q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
            const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel, 
            const uint16_t padding, const uint16_t stride, const q7_t *bias, 
            const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out, 
            const uint16_t dim_im_out, q15_t *bufferA, q7_t *bufferB){
        arm_convolve_HWC_q7_fast(Im_in,
                dim_im_in,
                ch_im_in,
                wt,
                ch_im_out,
                dim_kernel,
                padding,
                stride,
                bias,
                bias_shift,
                out_shift,
                Im_out,
                dim_im_out,
                bufferA,
                bufferB);

    }

};

struct CmsisConvolveRGB {
    void operator()(const q7_t *Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
            const q7_t *wt, const uint16_t ch_im_out, const uint16_t dim_kernel, 
            const uint16_t padding, const uint16_t stride, const q7_t *bias, 
            const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out, 
            const uint16_t dim_im_out, q15_t *bufferA, q7_t *bufferB){
        arm_convolve_HWC_q7_rgb(Im_in,
                dim_im_in,
                ch_im_in,
                wt,
                ch_im_out,
                dim_kernel,
                padding,
                stride,
                bias,
                bias_shift,
                out_shift,
                Im_out,
                dim_im_out,
                bufferA,
                bufferB);

    }

};


template<typename T, typename OP_SELECTOR>
void CmsisConvolution(S_TENSOR im_in_s, S_TENSOR wt_s, S_TENSOR padding_s,
                         S_TENSOR stride_s, S_TENSOR bias_s,
                         S_TENSOR bias_shift_s, S_TENSOR out_shift_s,
                         S_TENSOR im_out_s, S_TENSOR bufferA_s,
                         S_TENSOR bufferB_s, T byteme1, const OP_SELECTOR& op)
{
    const T* im_in  = im_in_s->read<T>(0, sizeof(T)); //Read one byte
    const T* wt = wt_s->read<T>(0, sizeof(T));
    const T* bias = bias_s->read<T>(0, sizeof(T));
    const uint16_t dim_im_in = im_in_s->getShape()[0];
    const uint16_t dim_im_out = im_out_s->getShape()[0];
    const uint16_t dim_kernel = wt_s->getShape()[0];
    const uint16_t ch_im_in = im_in_s->getShape()[2];
    const uint16_t ch_im_out = im_out_s->getShape()[2];
    const uint16_t padding = *(padding_s->read<uint16_t>(0,0));
    const uint16_t stride = *(stride_s->read<uint16_t>(0,0));
    const uint16_t bias_shift = *(bias_shift_s->read<uint16_t>(0,0));
    const uint16_t out_shift = *(out_shift_s->read<uint16_t>(0,0));
    
    T* im_out = im_out_s->write<T>(0, sizeof(T));
    q15_t* bufferA = bufferA_s->write<q15_t>(0, sizeof(q15_t));
    q7_t* bufferB = bufferB_s->write<q7_t>(0, sizeof(q15_t));

    op(im_in, dim_im_in, ch_im_in, wt, ch_im_out, dim_kernel,
       padding, stride, bias, bias_shift, out_shift, im_out, dim_im_out,
       bufferA, bufferB);
    //Error checking
}

// Default to cmsis basic convolve as these are guaranteed to work
template <class T, class TARGET_OP=CmsisConvolveBasic>
class ConvolutionCmsisOp : public Operator {
  public:
  FullyConnectedLayerCmsisOp() {
    n_inputs = 7;
    n_outputs = 3;
  }
  virtual void compute() override {
    T x;
    TARGET_OP op;
    TARGET_OP()(inputs[0], inputs[1], inputs[2], 
        inputs[3], inputs[4], 
        inputs[5], inputs[6],
        outputs[0], outputs[1], outputs[2], x, op);
  }
};


#endif /*UTENSOR_CMSIS_CONVOLUTION_HPP__*/
