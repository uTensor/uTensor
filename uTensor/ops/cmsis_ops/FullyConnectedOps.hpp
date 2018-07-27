#ifndef UTENSOR_FULLY_CONNECTED_OPS
#define UTENSOR_FULLY_CONNECTED_OPS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"

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
template<typename T1, typename T2, typename TOut>
void FullyConnectedLayerCmsis(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                         S_TENSOR bShift, S_TENSOR oShift,
                         S_TENSOR pOut, S_TENSOR scratch,
                         T1 byteme1, T2 byteme2, TOut byteme3)
{
    //Throw error if this gets called
}

template<>
void FullyConnectedLayerCmsis<q7_t, q7_t, q7_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                         q7_t byteme1, q7_t byteme2, q7_t byteme3)
{
    const q7_t* iV_data = iV->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* mW_data = mW->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* bias_data = b->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q7_t* pOut_data = pOut->write<q7_t>(0, sizeof(q7_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_q7(iV_data, mW_data, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    

    //Error checking
}
template<>
void FullyConnectedLayerCmsis<q15_t, q15_t, q15_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                         q15_t byteme1, q15_t byteme2, q15_t byteme3)
{
    const q15_t* iV_data = iV->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q15_t* mW_data = mW->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q15_t* bias_data = b->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q15_t* pOut_data = pOut->write<q15_t>(0, sizeof(q15_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_q15(iV_data, mW_data, dim_vec, num_of_rows, 
                        bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //Error checking
}

template<>
void FullyConnectedLayerCmsis<q15_t, q7_t, q7_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                         q15_t byteme1, q7_t byteme2, q7_t byteme3)
{
    const q15_t* iV_data = iV->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q7_t* mW_data = mW->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* bias_data = b->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q15_t* pOut_data = pOut->write<q15_t>(0, sizeof(q15_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_mat_q7_vec_q15(iV_data, mW_data, dim_vec, num_of_rows, 
                        bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //Error checking
}

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
template<typename T1, typename T2, typename TOut>
void FullyConnectedLayerOptCmsis(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                         S_TENSOR bShift, S_TENSOR oShift,
                         S_TENSOR pOut, S_TENSOR scratch,
                         T1 byteme1, T2 byteme2, TOut byteme3)
{
    //Throw error if this gets called
}

template<>
void FullyConnectedLayerOptCmsis<q7_t, q7_t, q7_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                                           q7_t byteme1, q7_t byteme2, q7_t byteme3)
{
    const q7_t* iV_data = iV->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* mW_data = mW->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* bias_data = b->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q7_t* pOut_data = pOut->write<q7_t>(0, sizeof(q7_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_q7_opt(iV_data, mW_data, dim_vec, num_of_rows, 
                        bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //Error checking
}
template<>
void FullyConnectedLayerOptCmsis<q15_t, q15_t, q15_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                                           q15_t byteme1, q15_t byteme2, q15_t byteme3)
{
    const q15_t* iV_data = iV->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q15_t* mW_data = mW->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q15_t* bias_data = b->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q15_t* pOut_data = pOut->write<q15_t>(0, sizeof(q15_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_q15_opt(iV_data, mW_data, dim_vec, num_of_rows, 
                        bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //Error checking
}

template<>
void FullyConnectedLayerOptCmsis<q15_t, q7_t, q7_t>(S_TENSOR iV, S_TENSOR mW, S_TENSOR b,
                                           S_TENSOR bShift, S_TENSOR oShift,
                                           S_TENSOR pOut, S_TENSOR scratch,
                                           q15_t byteme1, q7_t byteme2, q7_t byteme3)
{
    const q15_t* iV_data = iV->read<q15_t>(0, sizeof(q15_t)); //Read one byte
    const q7_t* mW_data = mW->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const q7_t* bias_data = b->read<q7_t>(0, sizeof(q7_t)); //Read one byte
    const uint16_t dim_vec = iV->getShape()[0];
    const uint16_t num_of_rows = mW->getShape()[0];
    const uint16_t bias_shift = *(bShift->read<uint16_t>(0,0));
    const uint16_t out_shift = *(oShift->read<uint16_t>(0,0));
    pOut->resize(b->getShape());
    q15_t* pOut_data = pOut->write<q15_t>(0, sizeof(q15_t));
    q15_t* scratch_data = scratch->write<q15_t>(0, sizeof(q15_t));

    arm_fully_connected_mat_q7_vec_q15_opt(iV_data, mW_data, dim_vec, num_of_rows, 
                        bias_shift, out_shift, bias_data, pOut_data, scratch_data);
    //Error checking
}


template <class T1, class T2, class TOut>
class FullyConnectedLayerCmsisOp : public Operator {
  public:
  FullyConnectedLayerCmsisOp() {
    n_inputs = 6;
    n_outputs = 1;
  }
  virtual void compute() override {
    T1 x;
    T2 y;
    TOut byteme;
    FullyConnectedLayerCmsis(inputs[0], inputs[1], inputs[2], inputs[3],
      inputs[4], outputs[0], inputs[5], x, y, byteme);
  }
};


template <class T1, class T2, class TOut>
class FullyConnectedLayerOptCmsisOp : public Operator {
  public:
  FullyConnectedLayerOptCmsisOp() {
    n_inputs = 6;
    n_outputs = 1;
  }
  virtual void compute() override {
    T1 x;
    T2 y;
    TOut byteme;
    FullyConnectedLayerOptCmsis<T1, T2, TOut>(inputs[0], inputs[1], inputs[2], inputs[3],
      inputs[4], outputs[0], inputs[5], x, y, byteme);
  }
};


#endif /*UTENSOR_FULLY_CONNECTED_OPS*/
