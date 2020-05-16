#include "constants_layer_2_dws_conv2d.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

static localCircularArenaAllocator<400000, uint32_t> ram_allocator;
static localCircularArenaAllocator<2048> meta_allocator;

TEST(LayerByLayer, DWSConv2D_2) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Tensor input = new RomTensor({1, 28, 28, 1}, flt, input_quant_img);
  PerTensorQuantizationParams in_params(in_zp, in_scale);
  input->set_quantization_params(in_params);
  Tensor filter = new RomTensor({1, 3, 3, 32}, i8, arr_filter);
  PerChannelQuantizationParams filter_params(filter_zps, filter_scales);
  filter->set_quantization_params(filter_params);
  Tensor bias = new RomTensor({32}, i32, arr_bias);
  PerChannelQuantizationParams bias_params(bias_zps, bias_scales);
  bias->set_quantization_params(bias_params);

  Tensor output = new RamTensor({1, 26, 26, 32}, i8);
  PerTensorQuantizationParams out_params(out_zp, out_scale);
  output->set_quantization_params(out_params);

  QuantizedDepthwiseSeparableConvOperator<int8_t> op(
      {1, 1}, VALID, 32, {1, 1}, TFLM::TfLiteFusedActivation::kTfLiteActRelu);

  op
      .set_inputs({
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::in, input},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::filter, filter},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::bias, bias},
      })
      .set_outputs(
          {{QuantizedDepthwiseSeparableConvOperator<int8_t>::out, output}})
      .eval();
  for (int i = 0; i < 21632; ++i) {
    int8_t got = static_cast<int8_t>(output(i));
    EXPECT_NEAR(got, ref_output[i], 1) << "Output: " << static_cast<int16_t>(got) << ", Reference: " << static_cast<int16_t>(ref_output[i]) << ", Difference: " << got - ref_output[i];
  }
}
