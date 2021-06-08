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

const float a_buffer[25] = {26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
const float b_buffer[25] = {3,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,  12, 13,
                            14, 16, 15, 25, 24, 23, 22, 21, 20, 19, 18, 17};

TEST(Arithmetic, AddOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, flt, a_buffer);
  Tensor b = new RomTensor({5, 5}, flt, b_buffer);
  Tensor c = new RamTensor({5, 5}, flt);

  AddOperator<float> add_AB;
  add_AB.set_inputs({{AddOperator<float>::a, a}, {AddOperator<float>::b, b}})
      .set_outputs({{AddOperator<float>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<float>(c(lin_index)),
              a_buffer[lin_index] + b_buffer[lin_index]);
  }
}

TEST(Arithmetic, SubOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, flt, a_buffer);
  Tensor b = new RomTensor({5, 5}, flt, b_buffer);
  Tensor c = new RamTensor({5, 5}, flt);

  SubOperator<float> sub_AB;
  // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // c}});
  sub_AB.set_inputs({{SubOperator<float>::a, a}, {SubOperator<float>::b, b}})
      .set_outputs({{SubOperator<float>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<float>(c(lin_index)),
              a_buffer[lin_index] - b_buffer[lin_index]);
  }
}

TEST(Arithmetic, MulOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, flt, a_buffer);
  Tensor b = new RomTensor({5, 5}, flt, b_buffer);
  Tensor c = new RamTensor({5, 5}, flt);
  MulOperator<float> mul_AB;
  mul_AB.set_inputs({{MulOperator<float>::a, a}, {MulOperator<float>::b, b}})
      .set_outputs({{MulOperator<float>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<float>(c(lin_index)),
              a_buffer[lin_index] * b_buffer[lin_index]);
  }
}

TEST(Arithmetic, DivOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor a = new RomTensor({5, 5}, flt, a_buffer);
  Tensor b = new RomTensor({5, 5}, flt, b_buffer);
  Tensor c = new RamTensor({5, 5}, flt);
  DivOperator<float> div_op;
  div_op.set_inputs({{DivOperator<float>::a, a}, {DivOperator<float>::b, b}})
      .set_outputs({{DivOperator<float>::c, c}})
      .eval();

  // Compare results
  TensorShape& c_shape = c->get_shape();
  for (int lin_index = 0; lin_index < c_shape.num_elems(); ++lin_index) {
    EXPECT_EQ(static_cast<float>(c(lin_index)),
              a_buffer[lin_index] / b_buffer[lin_index]);
  }
}
