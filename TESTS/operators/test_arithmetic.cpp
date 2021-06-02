#include <cstring>
#include <iostream>

#include "Arithmetic.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "gtest/gtest.h"
#include "uTensor/core/context.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

const uint16_t a_buffer[25] = {26, 27, 28, 29, 30, 31, 32, 33, 34,
                               35, 36, 37, 38, 39, 40, 41, 42, 43,
                               44, 45, 46, 47, 48, 49, 50};
const uint16_t b_buffer[25] = {3,  3,  4,  5,  6,  7,  8,  9,  5,
                               6,  7,  12, 13, 14, 16, 15, 25, 24,
                               23, 22, 21, 20, 19, 18, 17};

TEST(Arithmetic, AddOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, u16, a_buffer);
  Tensor b = new RomTensor({5, 5}, u16, b_buffer);
  Tensor c = new RamTensor({5, 5}, u16);

  AddOperator<uint16_t> add_AB;
  add_AB
      .set_inputs(
          {{AddOperator<uint16_t>::a, a}, {AddOperator<uint16_t>::b, b}})
      .set_outputs({{AddOperator<uint16_t>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<uint16_t>(c(lin_index)),
              a_buffer[lin_index] + b_buffer[lin_index]);
  }
}

TEST(Arithmetic, SubOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, u16, a_buffer);
  Tensor b = new RomTensor({5, 5}, u16, b_buffer);
  Tensor c = new RamTensor({5, 5}, u16);

  SubOperator<uint16_t> sub_AB;
  // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // c}});
  sub_AB
      .set_inputs(
          {{SubOperator<uint16_t>::a, a}, {SubOperator<uint16_t>::b, b}})
      .set_outputs({{SubOperator<uint16_t>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<uint16_t>(c(lin_index)),
              a_buffer[lin_index] - b_buffer[lin_index]);
  }
}

TEST(Arithmetic, MulOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, u16, a_buffer);
  Tensor b = new RomTensor({5, 5}, u16, b_buffer);
  Tensor c = new RamTensor({5, 5}, u16);
  MulOperator<uint16_t> mul_AB;
  mul_AB
      .set_inputs(
          {{MulOperator<uint16_t>::a, a}, {MulOperator<uint16_t>::b, b}})
      .set_outputs({{MulOperator<uint16_t>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<uint16_t>(c(lin_index)),
              a_buffer[lin_index] * b_buffer[lin_index]);
  }
}
