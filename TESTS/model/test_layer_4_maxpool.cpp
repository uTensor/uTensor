#include "constants_layer_4_maxpool.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

static localCircularArenaAllocator<2048> ram_allocator;
static localCircularArenaAllocator<2048> meta_allocator;

TEST(LayerByLayer, MaxPool2D_4) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  Tensor input = new RomTensor({1, 5, 5, 10}, i8, arr_input);
  PerTensorQuantizationParams in_params(in_zp, in_scale);
  input->set_quantization_params(in_params);

  Tensor output = new RamTensor({1, 2, 2, 10}, i8);
  PerTensorQuantizationParams out_params(-128, 0.006759230513125658);
  output->set_quantization_params(out_params);

  MaxPoolOperator<int8_t> op({2, 2}, {1, 2, 2, 1}, VALID);

  op.set_inputs({
                    {MaxPoolOperator<int8_t>::in, input},
                })
      .set_outputs({
          {MaxPoolOperator<int8_t>::out, output},
      })
      .eval();
  for (int i = 0; i < 40; ++i) {
    EXPECT_NEAR(static_cast<int8_t>(output(i)), ref_outupt[i], 1);
  }
}