#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_sq_tanh.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantTanhTest, sq_tanh_0) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<128*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_output = new RomTensor({ 128 }, i8, s_ref_output_sq_tanh_0);
  ref_output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_sq_tanh_0_zp, s_ref_output_sq_tanh_0_scale));
  Tensor output = new RamTensor({ 128 }, flt);
  Tensor input = new RomTensor({ 128 }, i8, s_ref_input_sq_tanh_0);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_sq_tanh_0_zp, s_ref_input_sq_tanh_0_scale));

  uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t> tanh_op;
  tanh_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_in, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_out, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 128; i++) {
  EXPECT_NEAR(static_cast<float>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantTanhTest, sq_tanh_1) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<128*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor output = new RamTensor({ 128 }, flt);
  Tensor ref_output = new RomTensor({ 128 }, i8, s_ref_output_sq_tanh_1);
  ref_output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_sq_tanh_1_zp, s_ref_output_sq_tanh_1_scale));
  Tensor input = new RomTensor({ 128 }, i8, s_ref_input_sq_tanh_1);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_sq_tanh_1_zp, s_ref_input_sq_tanh_1_scale));

  uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t> tanh_op;
  tanh_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_in, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_out, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 128; i++) {
  EXPECT_NEAR(static_cast<float>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantTanhTest, sq_tanh_2) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<128*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 128 }, i8, s_ref_input_sq_tanh_2);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_sq_tanh_2_zp, s_ref_input_sq_tanh_2_scale));
  Tensor ref_output = new RomTensor({ 128 }, i8, s_ref_output_sq_tanh_2);
  ref_output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_sq_tanh_2_zp, s_ref_output_sq_tanh_2_scale));
  Tensor output = new RamTensor({ 128 }, flt);

  uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t> tanh_op;
  tanh_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_in, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_out, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 128; i++) {
  EXPECT_NEAR(static_cast<float>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantTanhTest, sq_tanh_3) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<128*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 128 }, i8, s_ref_input_sq_tanh_3);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_sq_tanh_3_zp, s_ref_input_sq_tanh_3_scale));
  Tensor output = new RamTensor({ 128 }, flt);
  Tensor ref_output = new RomTensor({ 128 }, i8, s_ref_output_sq_tanh_3);
  ref_output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_sq_tanh_3_zp, s_ref_output_sq_tanh_3_scale));

  uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t> tanh_op;
  tanh_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_in, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_out, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 128; i++) {
  EXPECT_NEAR(static_cast<float>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantTanhTest, sq_tanh_4) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<128*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor output = new RamTensor({ 128 }, flt);
  Tensor ref_output = new RomTensor({ 128 }, i8, s_ref_output_sq_tanh_4);
  ref_output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_sq_tanh_4_zp, s_ref_output_sq_tanh_4_scale));
  Tensor input = new RomTensor({ 128 }, i8, s_ref_input_sq_tanh_4);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_sq_tanh_4_zp, s_ref_input_sq_tanh_4_scale));

  uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t> tanh_op;
  tanh_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_in, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::TanhOperator<int8_t,int8_t>::act_out, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 128; i++) {
  EXPECT_NEAR(static_cast<float>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

