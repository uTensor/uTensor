
#include <cstring>
#include <iostream>

#include "BufferTensor.hpp"
#include "Convolution.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "constants_convolution.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

#define DO_STRIDE_TESTS 1
/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_0_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_0_stride_1);
  Tensor out = new RamTensor({1, 28, 28, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_0_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_1_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_1_stride_1);
  Tensor out = new RamTensor({1, 28, 28, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_1_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_2_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_2_stride_1);
  Tensor out = new RamTensor({1, 28, 28, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_2_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_3_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_3_stride_1);
  Tensor out = new RamTensor({1, 28, 28, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_3_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_4_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<25088 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_4_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_4_stride_1);
  Tensor out = new RamTensor({1, 28, 28, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 25088; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_4_stride_1[i], 0.0001);
  }
}

// STRIDE TESTS
#ifdef DO_STRIDE_TESTS
/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_0_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_0_stride_2);
  Tensor out = new RamTensor({1, 14, 14, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_0_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_1_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_1_stride_2);
  Tensor out = new RamTensor({1, 14, 14, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_1_stride_2[i], 0.0001);
  }
}
/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_2_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_2_stride_2);
  Tensor out = new RamTensor({1, 14, 14, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_2_stride_2[i], 0.0001);
  }
}
/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_3_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_3_stride_2);
  Tensor out = new RamTensor({1, 14, 14, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_3_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_4_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6272 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_4_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_4_stride_2);
  Tensor out = new RamTensor({1, 14, 14, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, SAME);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 6272; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_4_stride_2[i], 0.0001);
  }
}

#endif

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_0_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_0_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 18432; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_0_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_0_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_0_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 4608; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_0_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_1_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_1_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 18432; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_1_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_1_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_1_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 4608; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_1_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_2_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_2_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 18432; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_2_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_2_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_2_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 4608; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_2_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_3_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_3_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 18432; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_3_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_3_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_3_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 4608; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_3_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_4_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18432 * 2 * sizeof(float), uint32_t>
      ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_4_stride_1);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_4_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 1, 1, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 18432; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_4_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(Convolution, random_inputs_VALID_4_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4608 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 1}, flt, s_in_VALID_4_stride_2);
  Tensor w = new RomTensor({5, 5, 1, 32}, flt, s_w_VALID_4_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 32}, flt);

  Conv2dOperator<float> conv_Aw({1, 2, 2, 1}, VALID);
  conv_Aw
      .set_inputs(
          {{Conv2dOperator<float>::in, in}, {Conv2dOperator<float>::filter, w}})
      .set_outputs({{Conv2dOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 4608; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_4_stride_2[i], 0.0001);
  }
}
