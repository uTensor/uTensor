
#include <cstring>
#include <iostream>

#include "Convolution.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"

#include "constants_avgpool.hpp"
using std::cout;
using std::endl;

using namespace uTensor;


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2028*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2028; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 507; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_0_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 432; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2028*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2028; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 507; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_1_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 432; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2028*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2028; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 507; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_2_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 432; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2028*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2028; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 507; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_3_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_3_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 432; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_3_kh_5_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_1_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_1_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,28,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_1_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_1_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,14,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_1_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_1_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,28,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_1_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_1_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,14,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 1,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_1_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,26,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2184; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<546*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,13,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 546; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2028*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,26,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2028; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,13,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 507; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,26,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_3_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_3_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,13,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 3,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_3_kw_5_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_1_stride_1);
  Tensor out = new RamTensor({ 1,24,28,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 2016; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_1_stride_2);
  Tensor out = new RamTensor({ 1,12,14,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,1}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 504; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_3_stride_1);
  Tensor out = new RamTensor({ 1,24,26,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1872; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_3_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<468*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_3_stride_2);
  Tensor out = new RamTensor({ 1,12,13,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,3}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 468; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_3_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_5_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_5_stride_1);
  Tensor out = new RamTensor({ 1,24,24,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,1,1,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 1728; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_5_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(AvgPool, random_inputs_VALID_4_kh_5_kw_5_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_4_kh_5_kw_5_stride_2);
  Tensor out = new RamTensor({ 1,12,12,3 }, flt);

  AvgPoolOperator<float> mxpool({ 5,5}, { 1,2,2,1}, VALID);
  mxpool
       .set_inputs({ {ConvOperator<float>::in, in} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 432; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_VALID_4_kh_5_kw_5_stride_2[i], 0.0001);
  }
}

