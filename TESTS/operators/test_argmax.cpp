#include <cstring>
#include <iostream>

#include "ArgMinMax.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "constants_argmax.hpp"
#include "uTensor/core/context.hpp"
#include "gtest/gtest.h"

using namespace uTensor;

using namespace uTensor::ReferenceOperators;
 
TEST(ArgMax, random_argmax_test) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<10 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Tensor input_tensor = new RomTensor({10, 5}, flt, random_input_arr);
  Tensor axis_tensor = new RomTensor({1}, u32, const_axis);
  Tensor output_tensor = new RamTensor({10}, u32);
  ArgMaxOperator<float> op;
  op.set_inputs({{ArgMaxOperator<float>::input, input_tensor},
                 {ArgMaxOperator<float>::axis, axis_tensor}})
      .set_outputs({{ArgMaxOperator<float>::output, output_tensor}})
      .eval();
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ((uint32_t)output_tensor(i), ref_output_arr[i]);
  }
}
