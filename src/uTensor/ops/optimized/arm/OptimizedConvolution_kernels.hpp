#ifndef _UTENSOR_OPTIMIZED_CONV_KERNELS_HPP
#define _UTENSOR_OPTIMIZED_CONV_KERNELS_HPP
#include "Convolution_kernels.hpp"
#include "arm_nnfunctions.h"
#include "symmetric_quantization_utils.hpp"

namespace uTensor{
namespace CMSIS {
// Borrowed from TFLM
// TODO move TFLM bits to dedicated file
// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
inline int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int total_padding =
      ((out_size - 1) * stride + effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = total_padding % 2;
  return total_padding / 2;
}

void CMSIS_Conv_kernel(const Tensor& input, const Tensor& filter, const Tensor& bias, Tensor& output, const Padding padding, const uint16_t (&strides)[4]) {

  AllocatorInterface* ram_allocator =
      Context::get_default_context()->get_ram_data_allocator();
    const TensorShape& filter_shape = filter->get_shape();
    const TensorShape& input_shape = input->get_shape();
    const TensorShape& output_shape = output->get_shape();
    
    const int16_t input_depth = input_shape[3];
    const int16_t input_rows  = input_shape[1];
    const int16_t input_cols  = input_shape[2];
    const int16_t input_batches = input_shape[0];
    const int16_t filter_rows = filter_shape[1];
    const int16_t filter_cols = filter_shape[2];
    const int16_t stride_rows = strides[1];
    const int16_t stride_cols = strides[2];
    const int16_t output_rows = output_shape[1];
    const int16_t output_cols = output_shape[2];

    const int num_output_channels = output->get_quantization_params().num_channels();


    int32_t output_activation_min;
    int32_t output_activation_max;
    TFLM::CalculateActivationRangeQuantized(
        TFLM::kTfLiteActNone, output, &output_activation_min, &output_activation_max);

    int offset;
    int pad_height = 
      ComputePaddingWithOffset(stride_rows, 1, input_rows,
                               filter_rows, out_rows, &offset);
    int pad_width = 
      ComputePaddingWithOffset(stride_cols, 1, input_cols,
                               filter_cols, out_cols, &offset);

    // Save RAM where we can, a little bit of computation is cheap
    // TODO eventuall this will be in codegen
    int32_t* output_shift = ram_allocator->allocate(sizeof(int32_t)*num_output_channels);
    int32_t* output_mult = ram_allocator->allocate(sizeof(int32_t)*num_output_channels);
    Handle output_shift_h(output_shift);
    Handle output_mult_h(output_mult);
    ram_allocator->bind(output_shift, &output_shift_h);
    ram_allocator->bind(output_mult, &output_mult_h);

    const int32_t* bias_data = nullptr;
    const int8_t* kernel_data = nullptr;
    const int8_t* input_data = nullptr;
    int8_t* output_data = nullptr;

    // Wrap in an if somehow
    const TensorShape& bias_shape = bias->get_shape();
    size_t bias_read = bias->get_readable_block(
        bias_data, bias_shape.get_linear_size(), 0);  // Attempt to read all of bias
    size_t kernel_read = filter->get_readable_block(
        kernel_data, filter_shape.get_linear_size(),
        0);  // Attempt to read all of kernel
    size_t input_read = input->get_readable_block(
        input_data, input_shape.get_linear_size(),
        0);  // Attempt to read all of input
    size_t output_read = output->get_writeable_block(
        output_data, output_shape.get_linear_size(),
        0);  // Attempt to read all of input
    // TODO replace the direct shape access with constants that are more clear
    q15_t* buffer_a =
        ram_allocator->allocate(
            arm_convolve_s8_get_buffer_size(input_shape[3], filter_shape[2],
                                            filter_shape[1]));
    // Make sure buffers dont get deallocated for temps
    Handle buffer_a_hndl(buffer_a);
    ram_allocator->bind(buffer_a, &buffer_a_hndl);

    arm_status arm_convolve_s8(
        input, input_shape[2], input_shape[1], input_shape[3], input_shape[0],
        kernel, filter_shape[0], filter_shape[2], filter_shape[1],
        pad_width, pad_height, _stride[1], _stride[0],
        bias_data, output_data, const int32_t* output_shift, const int32_t* output_mult,
        const int32_t out_offset, const int32_t input_offset,
        output_activation_min,
        output_activation_max, output_shape[1], output_shape[0],
        buffer_a);

    ram_allocator->unbind(buffer_a, &buffer_a_hndl);
    ram_allocator->unbind(output_mult, &output_mult_h);
    ram_allocator->unbind(output_shift, &output_shift_h);
    ram_allocator->deallocate(buffer_a);
    ram_allocator->deallocate(output_mult);
    ram_allocator->deallocate(output_shift);

}

void CMSIS_AvgPoolKernel(const Tensor& input, Tensor& output, uint16_t _k_size[2], uint16_t _stride[4], Padding _padding){

    const TensorShape& in_shape = input->get_shape();
    const TensorShape& out_shape = output->get_shape();
    const int16_t input_depth = in_shape[3];
    const int16_t input_rows = in_shape[1];
    const int16_t input_cols = in_shape[2];
    const int16_t input_batches = in_shape[0];
    const int16_t out_depth = input_depth;  // filter.out_channels();
    const int16_t filter_rows = _k_size[0];
    const int16_t filter_cols = _k_size[1];
    // const int16_t filter_count = filter.out_channels();

    const int16_t stride_rows = strides[1];
    const int16_t stride_cols = strides[2];

    // Compute for now, but should assume codegen does this
    int16_t out_rows = output->get_shape()[1];
    int16_t out_cols = output->get_shape()[2];
    if (padding == VALID) {
      // out_rows = (input_rows - filter_rows) / stride_rows + 1;
      // out_cols = (input_cols - filter_cols) / stride_cols + 1;
    } else {
      // SAME
      // out_rows = input_rows;
      // out_cols = input_cols;
    }
    int offset;
    int pad_height = 
      ComputePaddingWithOffset(stride_rows, 1, input_rows,
                               filter_rows, out_rows, &offset);
    int pad_width = 
      ComputePaddingWithOffset(stride_cols, 1, input_cols,
                               filter_cols, out_cols, &offset);
    // TODO check if this returns number of uint16_t elements, or number of bytes
    int32_t buff_size = arm_avgpool_s8_get_buffer_size(	out_cols, input_depth);
    uint16_t* bufferA = Context::get_default_context()->get_ram_data_allocator()->allocate(buff_size);
    int8_t* input_data = nullptr;
    int8_t* output_data = nullptr;
    size_t input_read = input->get_readable_block(
        input_data, in_shape.get_linear_size(),
        0);  // Attempt to read all of input
    size_t output_read = output->get_writeable_block(
        output_data, out_shape.get_linear_size(),
        0);  // Attempt to read all of input

    int32_t output_activation_min;
    int32_t output_activation_max;
    TFLM::CalculateActivationRangeQuantized(
        TFLM::kTfLiteActNone, output, &output_activation_min, &output_activation_max);

    arm_status result = arm_avgpool_s8(	
        input_rows,
        input_cols,
        out_rows,
        out_cols,
        strid_rows,
        stride_cols,
        filter_rows,
        filter_cols,
        pad_height,
        pad_width,
        output_activation_min,
        output_activation_max,
        input_depth,
        input_data,
        bufferA,
        output_data
        );
    Context::get_default_context()->get_ram_data_allocator()->deallocate(bufferA);
  }
void CMSIS_MaxPoolKernel(const Tensor& input, Tensor& output, uint16_t _k_size[2], uint16_t _stride[4], Padding _padding){

    const TensorShape& in_shape = input->get_shape();
    const TensorShape& out_shape = output->get_shape();
    const int16_t input_depth = in_shape[3];
    const int16_t input_rows = in_shape[1];
    const int16_t input_cols = in_shape[2];
    const int16_t input_batches = in_shape[0];
    const int16_t out_depth = input_depth;  // filter.out_channels();
    const int16_t filter_rows = _k_size[0];
    const int16_t filter_cols = _k_size[1];
    // const int16_t filter_count = filter.out_channels();

    const int16_t stride_rows = strides[1];
    const int16_t stride_cols = strides[2];

    // Compute for now, but should assume codegen does this
    int16_t out_rows = output->get_shape()[1];
    int16_t out_cols = output->get_shape()[2];
    if (padding == VALID) {
      // out_rows = (input_rows - filter_rows) / stride_rows + 1;
      // out_cols = (input_cols - filter_cols) / stride_cols + 1;
    } else {
      // SAME
      // out_rows = input_rows;
      // out_cols = input_cols;
    }
    int offset;
    int pad_height = 
      ComputePaddingWithOffset(stride_rows, 1, input_rows,
                               filter_rows, out_rows, &offset);
    int pad_width = 
      ComputePaddingWithOffset(stride_cols, 1, input_cols,
                               filter_cols, out_cols, &offset);
    // TODO check if this returns number of uint16_t elements, or number of bytes
    int32_t buff_size = arm_avgpool_s8_get_buffer_size(	out_cols, input_depth);
    //uint16_t* bufferA = Context::get_default_context()->get_ram_data_allocator()->allocate(buff_size);
    int8_t* input_data = nullptr;
    int8_t* output_data = nullptr;
    size_t input_read = input->get_readable_block(
        input_data, in_shape.get_linear_size(),
        0);  // Attempt to read all of input
    size_t output_read = output->get_writeable_block(
        output_data, out_shape.get_linear_size(),
        0);  // Attempt to read all of input

    int32_t output_activation_min;
    int32_t output_activation_max;
    TFLM::CalculateActivationRangeQuantized(
        TFLM::kTfLiteActNone, output, &output_activation_min, &output_activation_max);

    arm_status result = arm_max_pool_s8(	
        input_rows,
        input_cols,
        out_rows,
        out_cols,
        strid_rows,
        stride_cols,
        filter_rows,
        filter_cols,
        pad_height,
        pad_width,
        output_activation_min,
        output_activation_max,
        input_depth,
        input_data,
        NULL,
        output_data
        );
    //Context::get_default_context()->get_ram_data_allocator()->deallocate(bufferA);
  }
}
}
