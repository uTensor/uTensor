#include <cstring>
#include <iostream>

#include "BufferTensor.hpp"
#include "QuantizeOps.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
#include "quantizationPrimitives.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace TflmSymQuantOps;

const int8_t s_a[10] = {-33, -38, -5, -5, -49, -41, -95, 98, -36, 0};
float s_b[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float ref_b[10] = {-2.2243972, -2.8097649, 1.0536618, 1.0536618,
                         -4.0975738, -3.1609855, -9.482957, 13.112236,
                         -2.5756178, 1.6390295};
int offset = -14;
float scale = 0.11707353591918945f;

TEST(Quantization, DequantizeOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new /*const*/ RomTensor({10}, i8, s_a);
  a->set_quantization_params(PerTensorQuantizationParams(offset, scale));

  Tensor b = new /*const*/ BufferTensor({10}, flt, s_b);

  DequantizeOperator<float, int8_t> deq_A;
  
  deq_A
      .set_inputs({{DequantizeOperator<float,uint8_t>::a, a}})
      .set_outputs({{DequantizeOperator<float,uint8_t>::b, b}})
      .eval();

  // Compare results
  TensorShape& b_shape = b->get_shape();
  for (int i = 0; i < b_shape[0]; i++) {
    // Just need to cast the output
    // FIXME: rounding error due to simplified kernel and support functions
    EXPECT_LE((float)b(i) - ref_b[i], 1e-5f);
  }
}
