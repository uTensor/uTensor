#include <cstring>
#include <iostream>

#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "Transpose.hpp"
#include "arenaAllocator.hpp"
#include "constants_transpose.hpp"
#include "context.hpp"
#include "gtest/gtest.h"

using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;
TEST(Transpose, transpose_test) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<15 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor input_tensor = new RomTensor({3, 1, 5, 1}, flt, random_input_arr);
  Tensor perm_tensor = new RomTensor({4}, i32, transpose_perm_arr);

  TensorShape input_target_shape(3, 1, 5, 1);
  TensorShape input_shape = input_tensor->get_shape();
  EXPECT_TRUE(input_target_shape == input_shape);

  Tensor output_tensor = new RamTensor(flt);
  TransposeOperator<float> op;

  op.set_inputs({{TransposeOperator<float>::input, input_tensor},
                 {TransposeOperator<float>::perm, perm_tensor}})
      .set_outputs({{TransposeOperator<float>::output, output_tensor}})
      .eval();

  for (int i = 0; i < 15; ++i) {
    EXPECT_NEAR((float)output_tensor(i), ref_output_arr[i], 0.0001);
  }
  TensorShape target_shape(5, 1, 3, 1);
  TensorShape output_shape = output_tensor->get_shape();
  EXPECT_TRUE(target_shape == output_shape);
}
