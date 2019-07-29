#ifndef UTENSOR_FULLY_CONNECTED_OPS
#define UTENSOR_FULLY_CONNECTED_OPS
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"

/***
 * Here we have two types of kernel functions. The basic function implements the function using regular GEMV approach. The opt functions operates with weights in interleaved formats.
 *
 * http://www.keil.com/pack/doc/CMSIS/NN/html/group__FC.html#gae3857bb6375692e81dde8cbd70adec08
 */

//I definitely overcomplicated this part. Should just use f*** function overloading
void cmsis_fc_selector(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q7_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_q7(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

void cmsis_fc_selector(const q15_t* iV, const q15_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q15_t* bias, 
                       q15_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_q15(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

void cmsis_fc_selector(const q15_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q15_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_mat_q7_vec_q15(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

void cmsis_fc_selector_opt(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q7_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_q7_opt(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

void cmsis_fc_selector_opt(const q15_t* iV, const q15_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q15_t* bias, 
                       q15_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_q15_opt(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

void cmsis_fc_selector_opt(const q15_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       q15_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_mat_q7_vec_q15_opt(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

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
        T1* pOut_data = pOut->write<T1>(0, sizeof(T1));
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
        T1* pOut_data = pOut->write<T1>(0, sizeof(T1));
        q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));
    
        cmsis_fc_selector_opt(iV_data, mW_data, dim_vec, num_of_rows, 
                bias_shift, out_shift, bias_data, pOut_data, scratch_data);
        
    
        //Error checking
    }
};

//TARGET_OP={FullyConnectedLayerCmsis, FullyConnectedLayerOptCmsis}
template <class T1, class T2, class T3, class TARGET_OP=FullyConnectedLayerCmsis>
class FullyConnectedLayerCmsisOp : public Operator {
  public:
  FullyConnectedLayerCmsisOp() {
    n_inputs = 6;
    n_outputs = 1;
  }
  virtual void compute() override {
    T1 x;
    T2 y;
    T3 byteme;
    TARGET_OP()(inputs[0], inputs[1], inputs[2], inputs[3],
      inputs[4], outputs[0], inputs[5], x, y, byteme);
  }
};


#endif /*UTENSOR_FULLY_CONNECTED_OPS*/
