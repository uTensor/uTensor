
#include <cstring>
#include <iostream>

#include "constants_quantized_dws_conv.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace TflmSymQuantOps;

TEST(Quantized, reference_1_dws_conv) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21632 * 2 * sizeof(i8), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor A = new RomTensor({1, 28, 28, 1}, i8, s_ref_A);
  A->set_quantization_params(
      PerTensorQuantizationParams(s_ref_A_zp[0], s_ref_A_scale[0]));

  Tensor filter = new RomTensor({1, 3, 3, 32}, i8, s_ref_filter);
  filter->set_quantization_params(
      PerTensorQuantizationParams(s_ref_filter_zp[0], s_ref_filter_scale[0]));

  Tensor bias = new RomTensor({32}, i32, s_ref_bias);
  bias->set_quantization_params(
      PerTensorQuantizationParams(s_ref_bias_zp[0], s_ref_bias_scale[0]));

  Tensor output_ref = new RomTensor({1, 26, 26, 32}, i8, s_ref_output_ref);
  output_ref->set_quantization_params(PerTensorQuantizationParams(
      s_ref_output_ref_zp[0], s_ref_output_ref_scale[0]));

  Tensor out = new RamTensor({1, 26, 26, 32}, i8);
  out->set_quantization_params(PerTensorQuantizationParams(
      s_ref_output_ref_zp[0], s_ref_output_ref_scale[0]));

  /*
  options:
  {'Padding': 1, 'StrideW': 1, 'StrideH': 1, 'DepthMultiplier': 32,
  'FusedActivationFunction': '1 (RELU)', 'DilationWFactor': 1,
  'DilationHFactor': 1}
  */

  DepthwiseSeparableConvOperator<int8_t> dw_conv_Aw({1,1}, SAME, 32, {1, 1}, TFLM::TfLiteFusedActivation::kTfLiteActRelu );
  dw_conv_Aw
    .set_inputs({ {DepthwiseSeparableConvOperator<int8_t>::in, A}, {DepthwiseSeparableConvOperator<int8_t>::filter, filter}, {DepthwiseSeparableConvOperator<int8_t>::bias, bias} })
    .set_outputs({ {DepthwiseSeparableConvOperator<int8_t>::out, out} })
    .eval();

  //  for(int i = 0; i < out->get_shape().get_linear_size(); i++) {
  // 21632
  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(static_cast<int8_t>(out(i)), s_ref_output_ref[i], 1);
    // if(static_cast<int8_t>(out(i)) - s_ref_output_ref[i] != 0) {
    //   printf("out(%d): %d\n", i, static_cast<int8_t>(out(i)));
    //   printf("s_ref_output_ref[%d]: %d\n", i, s_ref_output_ref[i]);
    //   GTEST_FAIL();
    // }
  }
}
