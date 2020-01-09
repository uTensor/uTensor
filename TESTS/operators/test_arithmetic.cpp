#include "gtest/gtest.h"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "Arithmetic.hpp"
#include "BufferTensor.hpp"
#include "RomTensor.hpp"
#include "RamTensor.hpp"
#include <cstring>

#include <iostream>
using std::cout;
using std::endl;

using namespace uTensor; 


const uint8_t s_b[25] = {
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
  23,
  24};

TEST(Arithmetic, AddOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  uint8_t a_buffer[25];
  memset(a_buffer, 1, 25);
  //TensorInterface* a = new BufferTensor({5,5}, u8, a_buffer);
  //TensorInterface* b = new RomTensor({5,5}, u8, s_b);
  Tensor a = new BufferTensor({5,5}, u8, a_buffer);
  Tensor b = new RomTensor({5,5}, u8, s_b);
  Tensor c = new RamTensor({5,5}, u8);

  AddOperator<uint8_t> add_AB;
  add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a}, {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c, c}});
  TensorShape& c_shape = c->get_shape();
  for(int i = 0; i < c_shape[0]; i++){
    for(int j = 0; j < c_shape[1]; j++){
        size_t lin_index = i + j*c_shape[0];
        EXPECT_EQ((uint8_t)c(i,j), a_buffer[lin_index] + s_b[lin_index]);
    }
  }

}
