
#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_convolution_nobias.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_0_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_0_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 25088; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_0_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_0_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_0_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 6272; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_0_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_1_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 25088; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_1_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 6272; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_2_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_2_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 25088; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_2_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_SAME_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_SAME_2_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_SAME_2_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 6272; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_SAME_2_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_0_stride_1);
  Tensor out = new RamTensor({ 1,24,24,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 18432; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_0_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_0_stride_2);
  Tensor out = new RamTensor({ 1,12,12,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 4608; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_0_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_1_stride_1);
  Tensor out = new RamTensor({ 1,24,24,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 18432; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_1_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_1_stride_2);
  Tensor out = new RamTensor({ 1,12,12,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 4608; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_1_stride_2[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_stride_1);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_2_stride_1);
  Tensor out = new RamTensor({ 1,24,24,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,1,1,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 18432; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_stride_1[i], 0.0001);
  }
}


/*********************************************
 * Generated Test number 
 *********************************************/


TEST(ConvolutionNoBias, random_inputs_VALID_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,3 }, flt, s_in_VALID_2_stride_2);
  Tensor w = new RomTensor({ 32,5,5,3 }, flt, s_w_VALID_2_stride_2);
  Tensor out = new RamTensor({ 1,12,12,32 }, flt);

  Conv2dOperator<float> conv_Aw({ 1,2,2,1}, VALID);
  conv_Aw
     .set_inputs({ {Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w} })
     .set_outputs({ {Conv2dOperator<float>::out, out} })
     .eval();

  for(int i = 0; i < 4608; i++) {
  EXPECT_NEAR((float) out(i), s_ref_out_VALID_2_stride_2[i], 0.0001);
  }
}

