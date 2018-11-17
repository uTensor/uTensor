#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "FullyConnectedOps.hpp"

/***
 * Here we have two types of kernel functions. The basic function implements the function using regular GEMV approach. The opt functions operates with weights in interleaved formats.
 *
 * http://www.keil.com/pack/doc/CMSIS/NN/html/group__FC.html#gae3857bb6375692e81dde8cbd70adec08
 */

//I definitely overcomplicated this part. Should just use f*** function overloading
void cmsis_fc_selector(const q7_t* iV, const q7_t* mW, const uint16_t dim_vec, const uint16_t num_of_rows,
                       const uint16_t bias_shift, const uint16_t out_shift, const q7_t* bias, 
                       uint32_t* pOut, q15_t* scratch_data)
{
    arm_fully_connected_q7_tout(iV, mW, dim_vec, num_of_rows, 
            bias_shift, out_shift, bias, pOut, scratch_data);
}

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
