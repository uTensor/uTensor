#include "constants_quant_fully_connect_1.h"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

TEST(Quantization, QuantFullyConnectOp_1) {
  localCircularArenaAllocator<2048> meta_allocator;
  localCircularArenaAllocator<2048> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor input = new RomTensor({1, 5048}, i8, arr_input_1);
  PerTensorQuantizationParams param(input_zp_1, input_scale_1);
  input->set_quantization_params(param);

  Tensor filter = new RomTensor({5048, 128}, i8, arr_filter_1);
  Tensor bias = new RomTensor({128}, i8, arr_bias_1);

  Tensor output = new RamTensor({1, 128}, i8);

  QuantizedFullyConnectedOperator<int8_t> op(
      TFLM::TfLiteFusedActivation::kTfLiteActRelu);
  op.set_inputs({{QuantizedFullyConnectedOperator<int8_t>::input, input},
                 {QuantizedFullyConnectedOperator<int8_t>::filter, filter},
                 {QuantizedFullyConnectedOperator<int8_t>::bias, bias}})
      .set_outputs({{QuantizedFullyConnectedOperator<int8_t>::output, output}})
      .eval();

  for (int i = 0; i < output->num_elems(); ++i) {
    int8_t out_value = output(i);
    EXPECT_NEAR(out_value, ref_output_1[i], 5);
  }
}