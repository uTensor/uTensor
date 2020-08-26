#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_reduce_mean_axis_0_2.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantReduceMean, reduce_mean_axis_0_2_0) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_output = new RomTensor({ 3,7 }, i8, s_ref_output_reduce_mean_axis_0_2_0);
  Tensor output = new RamTensor({ 3,7 }, i8);
  output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_reduce_mean_axis_0_2_0_zp, s_ref_output_reduce_mean_axis_0_2_0_scale));
  Tensor input = new RomTensor({ 1,3,5,7 }, i8, s_ref_input_reduce_mean_axis_0_2_0);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_reduce_mean_axis_0_2_0_zp, s_ref_input_reduce_mean_axis_0_2_0_scale));

  uTensor::ReferenceOperators::ReduceMeanOperator<int8_t> mean_op({0, 2});
  mean_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::output, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<int8_t>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantReduceMean, reduce_mean_axis_0_2_1) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor output = new RamTensor({ 3,7 }, i8);
  output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_reduce_mean_axis_0_2_1_zp, s_ref_output_reduce_mean_axis_0_2_1_scale));
  Tensor ref_output = new RomTensor({ 3,7 }, i8, s_ref_output_reduce_mean_axis_0_2_1);
  Tensor input = new RomTensor({ 1,3,5,7 }, i8, s_ref_input_reduce_mean_axis_0_2_1);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_reduce_mean_axis_0_2_1_zp, s_ref_input_reduce_mean_axis_0_2_1_scale));

  uTensor::ReferenceOperators::ReduceMeanOperator<int8_t> mean_op({0, 2});
  mean_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::output, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<int8_t>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantReduceMean, reduce_mean_axis_0_2_2) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_output = new RomTensor({ 3,7 }, i8, s_ref_output_reduce_mean_axis_0_2_2);
  Tensor input = new RomTensor({ 1,3,5,7 }, i8, s_ref_input_reduce_mean_axis_0_2_2);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_reduce_mean_axis_0_2_2_zp, s_ref_input_reduce_mean_axis_0_2_2_scale));
  Tensor output = new RamTensor({ 3,7 }, i8);
  output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_reduce_mean_axis_0_2_2_zp, s_ref_output_reduce_mean_axis_0_2_2_scale));

  uTensor::ReferenceOperators::ReduceMeanOperator<int8_t> mean_op({0, 2});
  mean_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::output, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<int8_t>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantReduceMean, reduce_mean_axis_0_2_3) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor output = new RamTensor({ 3,7 }, i8);
  output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_reduce_mean_axis_0_2_3_zp, s_ref_output_reduce_mean_axis_0_2_3_scale));
  Tensor input = new RomTensor({ 1,3,5,7 }, i8, s_ref_input_reduce_mean_axis_0_2_3);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_reduce_mean_axis_0_2_3_zp, s_ref_input_reduce_mean_axis_0_2_3_scale));
  Tensor ref_output = new RomTensor({ 3,7 }, i8, s_ref_output_reduce_mean_axis_0_2_3);

  uTensor::ReferenceOperators::ReduceMeanOperator<int8_t> mean_op({0, 2});
  mean_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::output, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<int8_t>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(QuantReduceMean, reduce_mean_axis_0_2_4) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<21*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_output = new RomTensor({ 3,7 }, i8, s_ref_output_reduce_mean_axis_0_2_4);
  Tensor output = new RamTensor({ 3,7 }, i8);
  output->set_quantization_params(PerTensorQuantizationParams(s_ref_output_reduce_mean_axis_0_2_4_zp, s_ref_output_reduce_mean_axis_0_2_4_scale));
  Tensor input = new RomTensor({ 1,3,5,7 }, i8, s_ref_input_reduce_mean_axis_0_2_4);
  input->set_quantization_params(PerTensorQuantizationParams(s_ref_input_reduce_mean_axis_0_2_4_zp, s_ref_input_reduce_mean_axis_0_2_4_scale));

  uTensor::ReferenceOperators::ReduceMeanOperator<int8_t> mean_op({0, 2});
  mean_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::ReduceMeanOperator<int8_t>::output, output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<int8_t>( output(i) ), static_cast<int8_t>( ref_output(i) ), 2);
}
}

