
#include <cstring>
#include <iostream>

#include "BufferTensor.hpp"
#include "uTensor/ops/Convolution.hpp"
#include "RamTensor.hpp"
#include "RomTensor.hpp"
#include "arenaAllocator.hpp"
#include "constants_dws_convolution.hpp"
#include "uTensor/core/context.hpp"
#include "gtest/gtest.h"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_0_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5760 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_0_stride_1);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_0_stride_1);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_0_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 1, 1, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 5760; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_0_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_0_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_0_stride_2);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_0_stride_2);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_0_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 2, 2, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 1440; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_0_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_1_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5760 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_1_stride_1);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_1_stride_1);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_1_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 1, 1, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 5760; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_1_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_1_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_1_stride_2);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_1_stride_2);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_1_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 2, 2, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 1440; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_1_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_2_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5760 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_2_stride_1);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_2_stride_1);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_2_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 1, 1, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 5760; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_2_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_2_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_2_stride_2);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_2_stride_2);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_2_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 2, 2, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 1440; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_2_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_3_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5760 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_3_stride_1);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_3_stride_1);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_3_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 1, 1, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 5760; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_3_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_3_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_3_stride_2);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_3_stride_2);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_3_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 2, 2, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 1440; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_3_stride_2[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_4_stride_1) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5760 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_4_stride_1);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_4_stride_1);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_4_stride_1);
  Tensor out = new RamTensor({1, 24, 24, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 1, 1, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 5760; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_4_stride_1[i], 0.0001);
  }
}

/*********************************************
 * Generated Test number
 *********************************************/

TEST(DepthwiseSepConvolution, random_inputs_VALID_4_stride_2) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440 * 2 * sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  Tensor in = new RomTensor({1, 28, 28, 3}, flt, s_in_VALID_4_stride_2);
  Tensor dw = new RomTensor({5, 5, 3, 1}, flt, s_dw_VALID_4_stride_2);
  Tensor pw = new RomTensor({1, 1, 3, 10}, flt, s_pw_VALID_4_stride_2);
  Tensor out = new RamTensor({1, 12, 12, 10}, flt);

  DepthwiseSeparableConvOperator<float> dw_conv_Aw({1, 2, 2, 1}, VALID);
  dw_conv_Aw
      .set_inputs(
          {{DepthwiseSeparableConvOperator<float>::in, in},
           {DepthwiseSeparableConvOperator<float>::depthwise_filter, dw},
           {DepthwiseSeparableConvOperator<float>::pointwise_filter, pw}})
      .set_outputs({{DepthwiseSeparableConvOperator<float>::out, out}})
      .eval();

  for (int i = 0; i < 1440; i++) {
    EXPECT_NEAR((float)out(i), s_ref_out_VALID_4_stride_2[i], 0.0001);
  }
}
