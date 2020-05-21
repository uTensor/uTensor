#include "constants_layer_3_dws_conv2d.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

static localCircularArenaAllocator<2048> ram_allocator;
static localCircularArenaAllocator<2048> meta_allocator;

TEST(LayerByLayer, DWSConv2D_3) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  Tensor input = new RomTensor({1, 10, 10, 1}, i8, arr_input);
  PerTensorQuantizationParams in_params(in_zp, in_scale);
  input->set_quantization_params(in_params);

  Tensor filter = new RomTensor({1, 3, 3, 10}, i8, arr_filter);
  PerChannelQuantizationParams filter_params(filter_zps, filter_scales);
  filter->set_quantization_params(filter_params);

  Tensor bias = new RomTensor({10}, i32, arr_bias);
  PerChannelQuantizationParams bias_params(bias_zps, bias_scales);
  bias->set_quantization_params(bias_params);

  Tensor output = new RamTensor({1, 5, 5, 10}, i8);
  PerTensorQuantizationParams out_params(out_zp, out_scale);
  output->set_quantization_params(out_params);

  QuantizedDepthwiseSeparableConvOperator<int8_t> op({2, 2}, SAME, 10, {1, 1},
                                                     ::TFLM::kTfLiteActNone);
  op
      .set_inputs({
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::in, input},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::filter, filter},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::bias, bias},
      })
      .set_outputs({
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::out, output},
      })
      .eval();
  for (int i = 0; i < 250; ++i) {
    EXPECT_NEAR(static_cast<int8_t>(output(i)), ref_output[i], 1);
  }
}