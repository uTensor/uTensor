#ifndef UTENSOR_FULLY_CONNECTED_OPS
#define UTENSOR_FULLY_CONNECTED_OPS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"


#define ARM_MATH_DSP


template <class TOUT=q7_t>
arm_status

arm_fully_connected_q7_tout(const q7_t * pV,
                       const q7_t * pM,
                       const uint16_t dim_vec,
                       const uint16_t num_of_rows,
                       const uint16_t bias_shift,
                       const uint16_t out_shift, const q7_t * bias, TOUT * pOut, q15_t * vec_buffer)

{
#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    const q7_t *pB = pM;
    const q7_t *pB2;
    TOUT     *pO = pOut;
    const q7_t *pBias = bias;
    q15_t    *pA;
    uint16_t  rowCnt = num_of_rows >> 1;

    /* expand the vector into the buffer */
    arm_q7_to_q15_reordered_no_shift(pV, vec_buffer, dim_vec);

    while (rowCnt) {
    
        q31_t     sum = 0;
        q31_t     sum2 = 0;
        // q31_t     sum =  ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        // q31_t     sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;
        pB2 = pB + dim_vec;

        while (colCnt)
        {
            q31_t     inV, inM11, inM12, inM21, inM22;
            pB = (q7_t *) read_and_pad_reordered((void *)pB, &inM11, &inM12);
            pB2 = (q7_t *) read_and_pad_reordered((void *)pB2, &inM21, &inM22);

            inV = *__SIMD32(pA)++;

            sum = __SMLAD(inV, inM11, sum);
            sum2 = __SMLAD(inV, inM21, sum2);

            inV = *__SIMD32(pA)++;

            sum = __SMLAD(inV, inM12, sum);
            sum2 = __SMLAD(inV, inM22, sum2);

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t      inV = *pA++;
            q15_t     inM = *pB++;
            q15_t     inM2 = *pB2++;

            sum += inV * inM;
            sum2 += inV * inM2;
            colCnt--;
        }                       /* while over colCnt */

        // *pO++ = sum;
        // *pO++ = sum2;
        //added
        //*pO++ = (q7_t) (__SSAT((sum >> out_shift), 8));
        //*pO++ = (q7_t) (__SSAT((sum2 >> out_shift), 8));
        *pO++ = (TOUT) (__SSAT(((sum << 2 ) >> out_shift), (1 << 3) * sizeof(TOUT)));
        *pO++ = (TOUT) (__SSAT(((sum2 << 2 ) >> out_shift), (1 << 3) * sizeof(TOUT)));
        //*pO++ = (TOUT) (sum >> out_shift);
        //*pO++ = (TOUT) (sum2 >> out_shift);

        /* adjust the pointers and counters */
        pB += dim_vec;
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x1;

    while (rowCnt)
    {
        uint16_t  colCnt = dim_vec >> 2;
        //q31_t sum = 0
        q31_t     sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        pA = vec_buffer;

        while (colCnt)
        {
            q31_t     inV1, inV2, inM11, inM12;

            pB = (q7_t *) read_and_pad_reordered((void *)pB, &inM11, &inM12);

            inV1 = *__SIMD32(pA)++;
            sum = __SMLAD(inV1, inM11, sum);

            inV2 = *__SIMD32(pA)++;
            sum = __SMLAD(inV2, inM12, sum);

            colCnt--;
        }

        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t      inV = *pA++;
            q15_t     inM = *pB++;
            sum += inV * inM;
            colCnt--;
        }

        //*pO++ = sum;
        *pO++ = (TOUT) (__SSAT(((sum << 2) >> out_shift), (1 << 3) * sizeof(TOUT)));
        //*pO++ = (TOUT) (sum >> out_shift);

        rowCnt--;
    }

#else
    int       i, j;

    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    for (i = 0; i < num_of_rows; i++)
    {
        int       ip_out = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
        for (j = 0; j < dim_vec; j++)
        {
            ip_out += pV[j] * pM[i * dim_vec + j];
        }
        pOut[i] = (TOUT) ip_out;
        *pO++ = (TOUT) (__SSAT(((sum << 2 ) >> out_shift), (1 << 3) * sizeof(TOUT)));
        pOut[i] = (TOUT) (ip_out >> out_shift)
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to ARM_MATH_SUCCESS */
    return (ARM_MATH_SUCCESS);

}

/***
 * Here we have two types of kernel functions. The basic function implements the function using regular GEMV approach. The opt functions operates with weights in interleaved formats.
 *
 * http://www.keil.com/pack/doc/CMSIS/NN/html/group__FC.html#gae3857bb6375692e81dde8cbd70adec08
 */

//I definitely overcomplicated this part. Should just use f*** function overloading
void cmsis_fc_selector(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       uint32_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q7_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector(const q15_t* iV, const q15_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q15_t* bias, 
                       q15_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector(const q15_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q15_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector_opt(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q7_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector_opt(const q15_t* iV, const q15_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q15_t* bias, 
                       q15_t* pOut, q15_t* scratch_data);

void cmsis_fc_selector_opt(const q15_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q15_t* pOut, q15_t* scratch_data);

/**
 * @param [in] iV input vector
 * @param [in] mW matrix weights
 * @param [in] b bias
 * @param [in] scratch scrath computation space
 * @param [out] pOut output
 */
struct FullyConnectedLayerCmsis {
    template<typename T1, typename T2, typename T3>
    void operator() (S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                             S_TENSOR bShift, S_TENSOR oShift,
                             S_TENSOR pOut, S_TENSOR scratch,
                             T1 byteme1, T2 byteme2, T3 byteme3)
    {
        const T1* iV_data  = iV->read<T1>(0, sizeof(T1)); //Read one byte
        const T2* mW_data  = mW->read<T2>(0, sizeof(T2)); //Read one byte
        const T3* bias_data = b->read<T3>(0, sizeof(T3)); //Read one byte
        const uint16_t dim_vec = iV->getShape()[0];
        const uint16_t num_of_rows = mW->getShape()[0];
        const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
        const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
        pOut->resize(b->getShape());
        T1* pOut_data = pOut->write<T1>(0, sizeof(T1)); //FIXME: check the definiion of write()
        q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));
    
        cmsis_fc_selector(iV_data, mW_data, dim_vec, num_of_rows, 
                bias_shift, out_shift, bias_data, pOut_data, scratch_data);
        
    
        //Error checking
    }
};

/***
 * Here we have two types of kernel functions. The basic function implements the function using regular GEMV approach. The opt functions operates with weights in interleaved formats.
 *
 * http://www.keil.com/pack/doc/CMSIS/NN/html/group__FC.html#gae3857bb6375692e81dde8cbd70adec08
 */

/**
 * @param [in] iV input vector
 * @param [in] mW matrix weights
 * @param [in] b bias
 * @param [in] scratch scrath computation space
 * @param [out] pOut output
 */
struct FullyConnectedLayerOptCmsis{
    template<typename T1, typename T2, typename T3>
    void operator() (S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                             S_TENSOR bShift, S_TENSOR oShift,
                             S_TENSOR pOut, S_TENSOR scratch,
                             T1 byteme1, T2 byteme2, T3 byteme3)
    {
        const T1* iV_data  = iV->read<T1>(0, sizeof(T1)); //Read one byte
        const T2* mW_data  = mW->read<T2>(0, sizeof(T2)); //Read one byte
        const T3* bias_data = b->read<T3>(0, sizeof(T3)); //Read one byte
        const uint16_t dim_vec = iV->getShape()[0];
        const uint16_t num_of_rows = mW->getShape()[0];
        const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
        const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
        pOut->resize(b->getShape());
        T1* pOut_data = pOut->write<T1>(0, sizeof(T1));  //FIXME: check the definiion of write()
        q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));
    
        cmsis_fc_selector_opt(iV_data, mW_data, dim_vec, num_of_rows, 
                bias_shift, out_shift, bias_data, pOut_data, scratch_data);
        
    
        //Error checking
    }
};

// //TARGET_OP={FullyConnectedLayerCmsis, FullyConnectedLayerOptCmsis}
// template <class T1, class T2, class T3, class TARGET_OP=FullyConnectedLayerCmsis>
// class FullyConnectedLayerCmsisOp : public Operator {
//   public:
//   FullyConnectedLayerCmsisOp() {
//     n_inputs = 6;
//     n_outputs = 1;
//   }
//   virtual void compute() override {
//     T1 x;
//     T2 y;
//     T3 byteme;
//     TARGET_OP()(inputs[0], inputs[1], inputs[2], inputs[3],
//       inputs[4], outputs[0], inputs[5], x, y, byteme);
//   }
// };

template<class TOUT>
class FullyConnectedLayerCmsisOp : public Operator {
  public:
  FullyConnectedLayerCmsisOp() {
    n_inputs = 6;
    n_outputs = 1;
  }

  virtual void compute() override {
    S_TENSOR iV = inputs[0];
    S_TENSOR mW = inputs[1];
    S_TENSOR b = inputs[2];
    S_TENSOR bShift = inputs[3];
    S_TENSOR oShift = inputs[4];
    S_TENSOR pOut = outputs[0];
    S_TENSOR scratch = inputs[5];

    const q7_t* iV_data  = iV->read<q7_t>(0, 1); //Read one byte
    const q7_t* mW_data  = mW->read<q7_t>(0, 1); //Read one byte
    const q7_t* bias_data = b->read<q7_t>(0, 1); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0]; //W * v only
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    Shape outShape{num_of_rows, 1};
    pOut->resize(outShape);  //FIXME: compute the shape properly here
        TOUT* pOut_data = pOut->write<TOUT>(0, 1);
        q15_t* scratch_data = scratch->write<q15_t>(0, 1);
    
    printf("\r\n");
    // printf("FC: v dim: [%d, %d]", iV->getShape()[0], iV->getShape()[1]);
    // printf("FC: w dim: [%d, %d]", mW->getShape()[0], mW->getShape()[1]); 

    // printTensor<q7_t>(mW, 16);
    // printTensor<q7_t>(iV, 16);

    // printf("iV data: \r\n");
    // for(auto i=0; i < 16; i++) {
    //     printf("%d ", iV_data[i]);
    // }
    // printf("\r\n");
        arm_fully_connected_q7_tout(iV_data, mW_data, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //printTensor<int>(pOut, 16);
  }
};



#endif /*UTENSOR_FULLY_CONNECTED_OPS*/
