
#include <cstring>
#include <iostream>

#include "Convolution.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"

#include "constants_maxpool.hpp"
using std::cout;
using std::endl;

using namespace uTensor;


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 784; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<196*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 196; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<676*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 676; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 169; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<576*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 576; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_0_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_0_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 144; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 784; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<196*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 196; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<676*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 676; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 169; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<576*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 576; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_1_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_1_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 144; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 784; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<196*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 196; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<676*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 676; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 169; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<576*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 576; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_2_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_2_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 144; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 784; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<196*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 196; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<676*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 676; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 169; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<576*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 576; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_3_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_3_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 144; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 784; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<196*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 196; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 182; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<676*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 676; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 169; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<672*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 672; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 168; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<624*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 624; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 156; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<576*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 576; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(MaxPool, random_inputs_VALID_4_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_VALID_4_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,1 }, flt);

  MaxPoolOp<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 144; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_5_stride_2[i], 0.0001);
  }
}

