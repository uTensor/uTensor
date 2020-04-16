#include <cstring>
#include <iostream>

#include "arenaAllocator.hpp"
#include "context.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "ArgMinMax.hpp"

#include "gtest/gtest.h"

#include "constants_argmax.hpp"

using namespace uTensor;
 
TEST(ArgMax, random_argmax_test) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<10*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Tensor input_tensor = new RomTensor({ 10,5 }, flt, random_input_arr);
  Tensor axis_tensor = new RomTensor({ 1 }, u32, const_axis);
  Tensor output_tensor = new RamTensor({ 10 }, flt);
  ArgMaxOperator<float, float> op;
  op
    .set_inputs({ { ArgMaxOperator<float, float>::input, input_tensor }, { ArgMaxOperator<float, float>::axis, axis_tensor } })
    .set_outputs({ { ArgMaxOperator<float, float>::output, output_tensor } })
    .eval();
  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR((float) output_tensor(i), ref_output_arr[i], 0.0001);
  }
}