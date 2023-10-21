#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_rsqrt.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceRsqrt, random_gen_rsqrt__0) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4032*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 8,9,8,7 }, flt);
  Tensor out_ref = new RomTensor({ 8,9,8,7 }, flt, s_ref_out_0);
  Tensor input = new RomTensor({ 8,9,8,7 }, flt, s_ref_in_0);

  uTensor::ReferenceOperators::RsqrtOperator<float> rsqrt_op;
  rsqrt_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4032; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceRsqrt, random_gen_rsqrt__1) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<80*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 4,5,4 }, flt, s_ref_in_1);
  Tensor out = new RamTensor({ 4,5,4 }, flt);
  Tensor out_ref = new RomTensor({ 4,5,4 }, flt, s_ref_out_1);

  uTensor::ReferenceOperators::RsqrtOperator<float> rsqrt_op;
  rsqrt_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 80; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceRsqrt, random_gen_rsqrt__2) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<315*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 5,9,7 }, flt, s_ref_in_2);
  Tensor out_ref = new RomTensor({ 5,9,7 }, flt, s_ref_out_2);
  Tensor out = new RamTensor({ 5,9,7 }, flt);

  uTensor::ReferenceOperators::RsqrtOperator<float> rsqrt_op;
  rsqrt_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 315; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceRsqrt, random_gen_rsqrt__3) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<63*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor input = new RomTensor({ 7,9 }, flt, s_ref_in_3);
  Tensor out_ref = new RomTensor({ 7,9 }, flt, s_ref_out_3);
  Tensor out = new RamTensor({ 7,9 }, flt);

  uTensor::ReferenceOperators::RsqrtOperator<float> rsqrt_op;
  rsqrt_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 63; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceRsqrt, random_gen_rsqrt__4) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<540*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 6,10,9 }, flt);
  Tensor out_ref = new RomTensor({ 6,10,9 }, flt, s_ref_out_4);
  Tensor input = new RomTensor({ 6,10,9 }, flt, s_ref_in_4);

  uTensor::ReferenceOperators::RsqrtOperator<float> rsqrt_op;
  rsqrt_op
  .set_inputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::input, input }
  }).set_outputs({ 
    { uTensor::ReferenceOperators::RsqrtOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 540; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.001);
}
}

