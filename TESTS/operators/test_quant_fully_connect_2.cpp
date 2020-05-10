#include "constants_quant_fully_connect_2.h"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

TEST(Quantization, QuantFullyConnectOp_2) {
  localCircularArenaAllocator<2048> meta_allocator;
  localCircularArenaAllocator<2048> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor input = new RomTensor({1, 128}, i8, arr_input_2);
  PerTensorQuantizationParams param(input_zp_2, input_scale_2);
  input->set_quantization_params(param);

  Tensor filter = new RomTensor({128, 10}, i8, arr_filter_2);
  Tensor bias = new RomTensor({10}, i8, arr_bias_2);

  Tensor output = new RamTensor({1, 10}, i8);

  QuantizedFullyConnectedOperator<int8_t> op(
      TFLM::TfLiteFusedActivation::kTfLiteActNone);
  op.set_inputs({{QuantizedFullyConnectedOperator<int8_t>::input, input},
                 {QuantizedFullyConnectedOperator<int8_t>::filter, filter},
                 {QuantizedFullyConnectedOperator<int8_t>::bias, bias}})
      .set_outputs({{QuantizedFullyConnectedOperator<int8_t>::output, output}})
      .eval();

  for (int i = 0; i < output->num_elems(); ++i) {
    int8_t out_value = output(i);
    EXPECT_NEAR(out_value, ref_output_2[i], 5);
  }
}