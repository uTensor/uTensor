#include <cstring>
#include <iostream>

#include "StridedSlice.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;


using namespace uTensor;
using namespace uTensor::ReferenceOperators;



const uint8_t begin_buffer[3] = {1,  0,  0};

const uint8_t end_buffer[3] = {2, 1, 3};

const uint8_t strides_buffer[3] = {1,1,1};


TEST(Arithmetic, AddOp) {
  localCircularArenaAllocator<256> meta_allocator;
  localCircularArenaAllocator<256> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  uint8_t input_buffer[18];
  memset(input_buffer, 1, 18);


  Tensor i = new BufferTensor({2, 3, 3}, u8, input_buffer);
  Tensor b = new RomTensor({3}, u8, begin_buffer);
  Tensor e = new RomTensor({3}, u8, end_buffer);
  Tensor s = new RomTensor({3}, u8, strides_buffer);
  Tensor o = new RamTensor({1, 1, 3}, u8);

//   i, b, e, s, b_m, ell_m, e_m, n_a_m, s_a_m
  int b_m = 0;
  int ell_m = 0;
  int e_m = 0;
  int n_a_m = 0;
  int s_a_m = 0;

  StridedSliceOperator<uint8_t> stridedslice_AB;
  // add_AB.set_inputs(FixedTensorMap<2>({{AddOperator<uint8_t>::a, a},
  // {AddOperator<uint8_t>::b, b}})).set_outputs({{AddOperator<uint8_t>::c,
  // c}});

  stridedslice_AB.set_inputs({
          {StridedSliceOperator<uint8_t>::i, i}, 
          {StridedSliceOperator<uint8_t>::b, b},
          {StridedSliceOperator<uint8_t>::e, e},
          {StridedSliceOperator<uint8_t>::s, s},
          {StridedSliceOperator<uint8_t>::b_m, b_m},
          {StridedSliceOperator<uint8_t>::ell_m, ell_m},
          {StridedSliceOperator<uint8_t>::e_m, e_m},
          {StridedSliceOperator<uint8_t>::n_a_m, n_a_m},
          {StridedSliceOperator<uint8_t>::s_a_m, s_a_m}
        }).set_outputs({{StridedSliceOperator<uint8_t>::o, o}}).eval();

  // Compare results
  TensorShape& c_shape = o->get_shape();
  for (int i = 0; i < c_shape[0]; i++) {
    for (int j = 0; j < c_shape[1]; j++) {
        for (int k = 0; k < c_shape[2]; k++){
            cout << (uint8_t)o(i, j, k) << ", ";

        }
    }
  };


}
