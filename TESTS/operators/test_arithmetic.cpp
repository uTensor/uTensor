#include <cstring>
#include <iostream>

#include "Arithmetic.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "uTensor/core/context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;


using namespace uTensor;
using namespace uTensor::ReferenceOperators;

const uint8_t s_b[25] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

namespace uTensor {
  int get_size(TensorShape vec)
  {
      uint32_t size = 1;
      for (uint8_t i = 0; i<4; i++ ){
          if (vec[i] !=0){
              size *= vec[i];
          }
      }
      return size;
  }
}


TEST(Arithmetic, AddOp) {



  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  uint8_t a_buffer[25];
  memset(a_buffer, 1, 25);
  // std::fill_n(a_buffer, 1, 25);
  for (int i = 0; i < 25; i++){
        a_buffer[i] = 3;
        cout << int(a_buffer[i]) << ' ';
        // cout << i << " ";
    }
  // TensorInterface* a = new BufferTensor({5,5}, u8, a_buffer);
  // TensorInterface* b = new RomTensor({5,5}, u8, s_b);
  Tensor a = new BufferTensor({5, 5}, u8, a_buffer);
  Tensor b = new /*const*/ RomTensor({5, 5}, u8, s_b);
  Tensor c = new RamTensor({5, 5}, u8);
  TensorShape& c_shape = c->get_shape();
  int c_size = get_size(c_shape);

  // AddOperator<uint8_t> add_AB;
  // // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // // c}});
  // add_AB
  //     .set_inputs({{AddOperator<uint8_t>::a, a}, {AddOperator<uint8_t>::b, b}})
  //     .set_outputs({{AddOperator<uint8_t>::c, c}})
  //     .eval();

  // // Compare results
  // TensorShape& c_shape = c->get_shape();
  // for (int i = 0; i < c_shape[0]; i++) {
  //   for (int j = 0; j < c_shape[1]; j++) {
  //     size_t lin_index = j + i * c_shape[0];
  //     // Just need to cast the output
  //     EXPECT_EQ((uint8_t)c(i, j), a_buffer[lin_index] + s_b[lin_index]);
  //   }
  // }

  // SubOperator<int8_t> sub_AB;
  // // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // // c}});
  // sub_AB
  //     .set_inputs({{SubOperator<int8_t>::a, a}, {SubOperator<int8_t>::b, b}})
  //     .set_outputs({{SubOperator<uint8_t>::c, c}})
  //     .eval();

  // // Compare results
  // TensorShape& c_shape = c->get_shape();
  // for (int i = 0; i < c_shape[0]; i++) {
  //   for (int j = 0; j < c_shape[1]; j++) {
  //     size_t lin_index = j + i * c_shape[0];
  //     // Just need to cast the output
  //     cout << "c(i, j):" << int((int8_t)c(i, j)) << endl;
  //     cout << "a_buffer[lin_index]:" << int(a_buffer[lin_index]) << endl;
  //     cout << "s_b[lin_index]:" << int(s_b[lin_index]) << endl;

  //     EXPECT_EQ((int8_t)c(i, j), a_buffer[lin_index] - s_b[lin_index]);
  //   }
  // }


  MulOperator<uint8_t> mul_AB;
  // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // c}});
  mul_AB
      .set_inputs({{MulOperator<uint8_t>::a, a}, {MulOperator<uint8_t>::b, b}})
      .set_outputs({{MulOperator<uint8_t>::c, c}})
      .eval();

  // Compare results
  // TensorShape& c_shape = c->get_shape();
  for (int i = 0; i < c_shape[0]; i++) {
    for (int j = 0; j < c_shape[1]; j++) {
      size_t lin_index = j + i * c_shape[0];
      // Just need to cast the output
      cout << "c(i, j):" << int((uint8_t)c(i, j)) << endl;
      // cout << "size of c c:" << int(sizeof() << endl;
      cout << "a_buffer[lin_index]:" << int(a_buffer[lin_index]) << endl;
      cout << "s_b[lin_index]:" << int(s_b[lin_index]) << endl;
      cout << "try multiply:" << int(a_buffer[lin_index] * s_b[lin_index]) << endl;
      cout << "c_size:" << int(c_size) << endl;

      EXPECT_EQ(int((uint8_t)c(i, j)), int(a_buffer[lin_index] * s_b[lin_index]));
    }
  }
}

