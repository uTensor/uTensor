#include <cstring>
#include <iostream>

#include "BufferTensor.hpp"
#include "Matrix.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

const uint8_t s_a[4] = {1, 2, 3, 4};
const uint8_t s_b[4] = {5, 6, 7, 8};
const uint8_t s_c_ref[4] = {19, 22, 43, 50};

TEST(Matrix, MultSquareOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new /*const*/ RomTensor({2, 2}, u8, s_a);
  Tensor b = new /*const*/ RomTensor({2, 2}, u8, s_b);
  Tensor c = new RamTensor({2, 2}, u8);

  MatrixMultOperator<uint8_t> mult_AB;
  // add_AB.set_inputs(FixedTensorMap<2>({{MatrixMultOperator<uint8_t>::a, a},
  // {MatrixMultOperator<uint8_t>::b,
  // b}})).set_outputs({{MatrixMultOperator<uint8_t>::c, c}});
  mult_AB
      .set_inputs({{MatrixMultOperator<uint8_t>::a, a},
                   {MatrixMultOperator<uint8_t>::b, b}})
      .set_outputs({{MatrixMultOperator<uint8_t>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int i = 0; i < c_shape[0]; i++) {
    for (int j = 0; j < c_shape[1]; j++) {
      size_t lin_index = j + i * c_shape[0];
      // Just need to cast the output
      EXPECT_EQ((uint8_t)c(i, j), s_c_ref[lin_index]);
    }
  }
}
