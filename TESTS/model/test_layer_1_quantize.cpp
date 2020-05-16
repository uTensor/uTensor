#include "constants_layer_1_quantize.hpp"
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;

static localCircularArenaAllocator<2048> ram_allocator;
static localCircularArenaAllocator<2048> meta_allocator;

TEST(LayerByLayer, Quantize) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Tensor input = new RomTensor({1, 28, 28, 1}, flt, arr_image);
  Tensor output = new RamTensor({1, 28, 28, 1}, i8);
  PerTensorQuantizationParams params(zp, scale);
  output->set_quantization_params(params);
  ::TFLM::QuantizeOperator<int8_t, float> op;

  op.set_inputs({
                    {::TFLM::QuantizeOperator<int8_t, float>::input, input},
                })
      .set_outputs({
          {::TFLM::QuantizeOperator<int8_t, float>::output, output},
      })
      .eval();
  for (int i = 0; i < 784; ++i) {
    EXPECT_NEAR(static_cast<int8_t>(output(i)), ref_quant_img[i], 1);
  }
}