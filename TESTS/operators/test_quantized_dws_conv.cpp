
#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_quantized_dws_conv.hpp"
using std::cout;
using std::endl;

using namespace uTensor;



TEST(Quantized, reference_1_dws_conv) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21632*2*sizeof(i8), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  
  Tensor A =  new RomTensor({ 1, 28, 28, 1 }, i8, s_ref_A);
                        A->set_quantization_params(PerTensorQuantizationParams(s_ref_A_zp[0], s_ref_A_scale[0]));

  
  Tensor filter =  new RomTensor({ 1, 3, 3, 32 }, i8, s_ref_filter);
                        filter->set_quantization_params(PerTensorQuantizationParams(s_ref_filter_zp[0], s_ref_filter_scale[0]));

  
  Tensor bias =  new RomTensor({ 32 }, i32, s_ref_bias);
                        bias->set_quantization_params(PerTensorQuantizationParams(s_ref_bias_zp[0], s_ref_bias_scale[0]));

  
  Tensor output_ref =  new RomTensor({ 1, 26, 26, 32 }, i8, s_ref_output_ref);
                        output_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_output_ref_zp[0], s_ref_output_ref_scale[0]));

  
  Tensor out = new RamTensor({ 1, 26, 26, 32 }, i8);
                out->set_quantization_params(PerTensorQuantizationParams(s_ref_output_ref_zp[0], s_ref_output_ref_scale[0] ));

  TFLM::DepthwiseSeparableConvOperator<float> dw_conv_Aw;
  dw_conv_Aw
       .set_inputs({ {DepthwiseSeparableConvOperator<float>::in, A}, {DepthwiseSeparableConvOperator<float>::depthwise_filter, filter}, {DepthwiseSeparableConvOperator<float>::pointwise_filter, bias} })
       .set_outputs({ {DepthwiseSeparableConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < out->get_shape().get_linear_size(); i++) {
    EXPECT_NEAR((float) out(i), s_ref_output_ref[i], 0.0001);
  }
}
