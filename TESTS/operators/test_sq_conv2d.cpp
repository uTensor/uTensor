#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_sq_conv2d.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(Conv2D, random_gen_conv2d__0_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<9216*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,12,12,64 }, flt);
  Tensor in = new RomTensor({ 1,14,14,32 }, flt, s_ref_in_0_f);
  Tensor w = new RomTensor({ 64,3,3,32 }, flt, s_ref_w_0_f);
  Tensor b = new RomTensor({ 64 }, flt, s_ref_b_0_f);
  Tensor out_ref = new RomTensor({ 1,12,12,64 }, flt, s_ref_out_0_f);

  uTensor::ReferenceOperators::Conv2dOperator<float> convOp({1,1,1,1}, VALID);
  convOp
  .set_inputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<float>::in, in },
    { uTensor::ReferenceOperators::Conv2dOperator<float>::filter, w },
    { uTensor::ReferenceOperators::Conv2dOperator<float>::bias, b }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 9216; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Conv2D, random_gen_conv2d__0_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<9216*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,12,12,64 }, i8);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_0_q_zp, s_out_0_q_scale));
  Tensor in = new RomTensor({ 1,14,14,32 }, i8, s_ref_in_0_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_0_f_zp, s_ref_in_0_f_scale));
  Tensor w = new RomTensor({ 64,3,3,32 }, i8, s_ref_w_0_q);
  w->set_quantization_params(PerChannelQuantizationParams(s_ref_w_0_f_zp, s_ref_w_0_f_scale));
  Tensor b = new RomTensor({ 64 }, i32, s_ref_b_0_q);
  b->set_quantization_params(PerChannelQuantizationParams(s_ref_b_0_f_zp, s_ref_b_0_f_scale));
  Tensor out_ref = new RomTensor({ 1,12,12,64 }, i8, s_ref_out_0_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_0_f_zp, s_ref_out_0_f_scale));

  uTensor::ReferenceOperators::Conv2dOperator<int8_t> convOp({1,1}, VALID);
  convOp
  .set_inputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::in, in },
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::filter, w },
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::bias, b }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 9216; i++) {
  EXPECT_NEAR(static_cast<int8_t>( out(i) ), static_cast<int8_t>( out_ref(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Conv2D, random_gen_conv2d__1_f) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<9216*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,14,14,32 }, flt, s_ref_in_1_f);
  Tensor out_ref = new RomTensor({ 1,12,12,64 }, flt, s_ref_out_1_f);
  Tensor b = new RomTensor({ 64 }, flt, s_ref_b_1_f);
  Tensor w = new RomTensor({ 64,3,3,32 }, flt, s_ref_w_1_f);
  Tensor out = new RamTensor({ 1,12,12,64 }, flt);

  uTensor::ReferenceOperators::Conv2dOperator<float> convOp({1,1,1,1}, VALID);
  convOp
  .set_inputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<float>::in, in },
    { uTensor::ReferenceOperators::Conv2dOperator<float>::filter, w },
    { uTensor::ReferenceOperators::Conv2dOperator<float>::bias, b }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 9216; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(Conv2D, random_gen_conv2d__1_q) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<9216*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,14,14,32 }, i8, s_ref_in_1_q);
  in->set_quantization_params(PerTensorQuantizationParams(s_ref_in_1_f_zp, s_ref_in_1_f_scale));
  Tensor out_ref = new RomTensor({ 1,12,12,64 }, i8, s_ref_out_1_q);
  out_ref->set_quantization_params(PerTensorQuantizationParams(s_ref_out_1_f_zp, s_ref_out_1_f_scale));
  Tensor b = new RomTensor({ 64 }, i32, s_ref_b_1_q);
  b->set_quantization_params(PerChannelQuantizationParams(s_ref_b_1_f_zp, s_ref_b_1_f_scale));
  Tensor w = new RomTensor({ 64,3,3,32 }, i8, s_ref_w_1_q);
  w->set_quantization_params(PerChannelQuantizationParams(s_ref_w_1_f_zp, s_ref_w_1_f_scale));
  Tensor out = new RamTensor({ 1,12,12,64 }, i8);
  out->set_quantization_params(PerTensorQuantizationParams(s_out_1_q_zp, s_out_1_q_scale));

  uTensor::ReferenceOperators::Conv2dOperator<int8_t> convOp({1,1}, VALID);
  convOp
  .set_inputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::in, in },
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::filter, w },
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::bias, b }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::Conv2dOperator<int8_t>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 9216; i++) {
  EXPECT_NEAR(static_cast<int8_t>( out(i) ), static_cast<int8_t>( out_ref(i) ), 2);
}
}

