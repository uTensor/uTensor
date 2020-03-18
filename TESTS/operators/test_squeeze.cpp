
#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_squeeze.hpp"
using std::cout;
using std::endl;

using namespace uTensor;


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_float_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, flt);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_float_0[i];
  }

  SqueezeOperator<float> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<float>::x, io}})
       .eval();

  float tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_float_0[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_float_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, flt);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_float_1[i];
  }

  SqueezeOperator<float> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<float>::x, io}})
       .eval();

  float tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_float_1[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_float_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, flt);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_float_2[i];
  }

  SqueezeOperator<float> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<float>::x, io}})
       .eval();

  float tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_float_2[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_float_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, flt);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_float_3[i];
  }

  SqueezeOperator<float> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<float>::x, io}})
       .eval();

  float tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_float_3[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_float_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, flt);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_float_4[i];
  }

  SqueezeOperator<float> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<float>::x, io}})
       .eval();

  float tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_float_4[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int8_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i8);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int8_t_0[i];
  }

  SqueezeOperator<int8_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int8_t>::x, io}})
       .eval();

  int8_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int8_t_0[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int8_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i8);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int8_t_1[i];
  }

  SqueezeOperator<int8_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int8_t>::x, io}})
       .eval();

  int8_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int8_t_1[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int8_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i8);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int8_t_2[i];
  }

  SqueezeOperator<int8_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int8_t>::x, io}})
       .eval();

  int8_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int8_t_2[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int8_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i8);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int8_t_3[i];
  }

  SqueezeOperator<int8_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int8_t>::x, io}})
       .eval();

  int8_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int8_t_3[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int8_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int8_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i8);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int8_t_4[i];
  }

  SqueezeOperator<int8_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int8_t>::x, io}})
       .eval();

  int8_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int8_t_4[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int16_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int16_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i16);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int16_t_0[i];
  }

  SqueezeOperator<int16_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int16_t>::x, io}})
       .eval();

  int16_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int16_t_0[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int16_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int16_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i16);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int16_t_1[i];
  }

  SqueezeOperator<int16_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int16_t>::x, io}})
       .eval();

  int16_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int16_t_1[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int16_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int16_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i16);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int16_t_2[i];
  }

  SqueezeOperator<int16_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int16_t>::x, io}})
       .eval();

  int16_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int16_t_2[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int16_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int16_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i16);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int16_t_3[i];
  }

  SqueezeOperator<int16_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int16_t>::x, io}})
       .eval();

  int16_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int16_t_3[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int16_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int16_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i16);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int16_t_4[i];
  }

  SqueezeOperator<int16_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int16_t>::x, io}})
       .eval();

  int16_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int16_t_4[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int32_t_0) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int32_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i32);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int32_t_0[i];
  }

  SqueezeOperator<int32_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int32_t>::x, io}})
       .eval();

  int32_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int32_t_0[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int32_t_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int32_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i32);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int32_t_1[i];
  }

  SqueezeOperator<int32_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int32_t>::x, io}})
       .eval();

  int32_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int32_t_1[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int32_t_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int32_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i32);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int32_t_2[i];
  }

  SqueezeOperator<int32_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int32_t>::x, io}})
       .eval();

  int32_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int32_t_2[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int32_t_3) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int32_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i32);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int32_t_3[i];
  }

  SqueezeOperator<int32_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int32_t>::x, io}})
       .eval();

  int32_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int32_t_3[i]), 0, 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Squeeze, random_inputs_int32_t_4) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<64*2*sizeof(int32_t), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor io = new RamTensor({ 1,8,8,1 }, i32);
  for(int i = 0; i < 64; i++) {
    io(i) = s_in_int32_t_4[i];
  }

  SqueezeOperator<int32_t> Squeeze;
  Squeeze
       .set_inputs({ {SqueezeOperator<int32_t>::x, io}})
       .eval();

  int32_t tmp;
  for(int i = 0; i < 64; i++) {
    tmp = io(i);
    EXPECT_NEAR( (float)(tmp - s_ref_out_int32_t_4[i]), 0, 0.0001);
  }
}

