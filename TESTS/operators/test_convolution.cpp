
#include <cstring>
#include <iostream>

#include "Convolution.hpp"
#include "BufferTensor.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "context.hpp"
#include "gtest/gtest.h"

#include "constants_convolution.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

//#define DO_STRIDE_TESTS 1
/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_0_stride_1);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_0_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_0_stride_1[i], 0.0001);
  }
}




/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_1_stride_1);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_1_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_1_stride_1[i], 0.0001);
  }
}




/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float)> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_2_stride_1);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_2_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_2_stride_1[i], 0.0001);
  }
}




/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float)> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_3_stride_1);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_3_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_3_stride_1[i], 0.0001);
  }
}




/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_4_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_4_stride_1);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_4_stride_1);
  Tensor out = new RamTensor({ 1,28,28,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,1,1,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_4_stride_1[i], 0.0001);
  }
}

// STRIDE TESTS
#ifdef DO_STRIDE_TESTS
/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_0_stride_2);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_0_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_0_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_1_stride_2);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_1_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_1_stride_2[i], 0.0001);
  }
}
/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float)> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_2_stride_2);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_2_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_2_stride_2[i], 0.0001);
  }
}
/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float)> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_3_stride_2);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_3_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_3_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number 
 *********************************************/


TEST(Convolution, random_inputs_4_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272*2*sizeof(float), uint32_t> ram_allocator;
  Context::set_metadata_allocator(&meta_allocator);
  Context::set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({ 1,28,28,1 }, flt, s_in_4_stride_2);
  Tensor w = new RomTensor({ 5,5,1,32 }, flt, s_w_4_stride_2);
  Tensor out = new RamTensor({ 1,14,14,32 }, flt);

  ConvOperator<float> conv_Aw({ 1,2,2,1}, SAME);
  conv_Aw
       .set_inputs({ {ConvOperator<float>::in, in}, {ConvOperator<float>::filter, w} })
       .set_outputs({ {ConvOperator<float>::out, out} })
       .eval();

  for(int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float) out(i), s_ref_out_4_stride_2[i], 0.0001);
  }
}
#endif
