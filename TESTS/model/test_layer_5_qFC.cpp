#include "constants_layer_5_qFC.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

static localCircularArenaAllocator<4000> ram_allocator;
static localCircularArenaAllocator<2048> meta_allocator;

TEST(LayerByLayer, qFC_5) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  Tensor input = new RomTensor({1, 128}, i8, arr_input);
  PerTensorQuantizationParams in_params(in_zp, in_scale);
  input->set_quantization_params(in_params);
  Tensor filter = new RomTensor({128, 10}, i8, arr_filter);
  PerTensorQuantizationParams filter_params(filter_zp, filter_scale);
  filter->set_quantization_params(filter_params);
  Tensor bias = new RomTensor({10}, i32, arr_bias);
  PerTensorQuantizationParams bias_params(bias_zp, bias_scale);
  bias->set_quantization_params(bias_params);

  Tensor output = new RamTensor({1, 10}, i8);
  PerTensorQuantizationParams out_params(out_zp, out_scale);
  output->set_quantization_params(out_params);

  QuantizedFullyConnectedOperator<int8_t> op(TFLM::kTfLiteActNone);

  op.set_inputs({
                    {QuantizedFullyConnectedOperator<int8_t>::input, input},
                    {QuantizedFullyConnectedOperator<int8_t>::filter, filter},
                    {QuantizedFullyConnectedOperator<int8_t>::bias, bias},
                })
      .set_outputs({{QuantizedFullyConnectedOperator<int8_t>::output, output}})
      .eval();
  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(static_cast<int8_t>(output(i)), ref_output[i], 1);
  }
}