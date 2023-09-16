#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_sum.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceSum, random_gen_reduce_sum__0) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<108*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 3,9,7,4 }, flt, s_ref_in_0);
  Tensor axis = new RomTensor({ 1 }, i32, s_ref_axis_0);
  Tensor out = new RamTensor({ 3,9,4 }, flt);
  Tensor out_ref = new RomTensor({ 3,9,4 }, flt, s_ref_out_0);

  uTensor::ReferenceOperators::SumOperator<float> sum_op;
  sum_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::input, input },
    { uTensor::ReferenceOperators::SumOperator<float>::axis, axis }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 108; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceSum, random_gen_reduce_sum__1) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<112*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, s_ref_axis_1);
  Tensor out_ref = new RomTensor({ 7,4,4 }, flt, s_ref_out_1);
  Tensor input = new RomTensor({ 4,7,4,4 }, flt, s_ref_in_1);
  Tensor out = new RamTensor({ 7,4,4 }, flt);

  uTensor::ReferenceOperators::SumOperator<float> sum_op;
  sum_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::input, input },
    { uTensor::ReferenceOperators::SumOperator<float>::axis, axis }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 112; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceSum, random_gen_reduce_sum__2) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<360*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, s_ref_axis_2);
  Tensor out = new RamTensor({ 9,8,5 }, flt);
  Tensor out_ref = new RomTensor({ 9,8,5 }, flt, s_ref_out_2);
  Tensor input = new RomTensor({ 9,8,5,10 }, flt, s_ref_in_2);

  uTensor::ReferenceOperators::SumOperator<float> sum_op;
  sum_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::input, input },
    { uTensor::ReferenceOperators::SumOperator<float>::axis, axis }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 360; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceSum, random_gen_reduce_sum__3) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<3*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 3 }, flt);
  Tensor input = new RomTensor({ 5,3 }, flt, s_ref_in_3);
  Tensor out_ref = new RomTensor({ 3 }, flt, s_ref_out_3);
  Tensor axis = new RomTensor({ 1 }, i32, s_ref_axis_3);

  uTensor::ReferenceOperators::SumOperator<float> sum_op;
  sum_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::input, input },
    { uTensor::ReferenceOperators::SumOperator<float>::axis, axis }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 3; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceSum, random_gen_reduce_sum__4) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<30*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, s_ref_axis_4);
  Tensor out = new RamTensor({ 5,6 }, flt);
  Tensor input = new RomTensor({ 10,5,6 }, flt, s_ref_in_4);
  Tensor out_ref = new RomTensor({ 5,6 }, flt, s_ref_out_4);

  uTensor::ReferenceOperators::SumOperator<float> sum_op;
  sum_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::input, input },
    { uTensor::ReferenceOperators::SumOperator<float>::axis, axis }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::SumOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 30; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

