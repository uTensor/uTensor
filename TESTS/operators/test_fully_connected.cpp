#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_fully_connected.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceFC, random_gen_fc__0) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<512*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,512 }, flt, s_ref_out_0);
  Tensor b  = new RomTensor({ 512 }, flt, s_ref_b_0);
  Tensor w = new RomTensor({ 256,512 }, flt, s_ref_w_0);
  Tensor out = new RamTensor({ 1,512 }, flt);
  Tensor in = new RomTensor({ 1,256 }, flt, s_ref_in_0);

  FullyConnectedOperator<float> fcOp(Fuseable::NoActivation<float>);
  fcOp
  .set_inputs({ 
    { FullyConnectedOperator<float>::input, in },
    { FullyConnectedOperator<float>::filter, w },
    { FullyConnectedOperator<float>::bias, b  }
  }).set_outputs({ 
    { FullyConnectedOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceFC, random_gen_fc__1) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<512*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 512 }, flt, s_ref_b_1);
  Tensor out_ref = new RomTensor({ 1,512 }, flt, s_ref_out_1);
  Tensor w = new RomTensor({ 256,512 }, flt, s_ref_w_1);
  Tensor out = new RamTensor({ 1,512 }, flt);
  Tensor in = new RomTensor({ 1,256 }, flt, s_ref_in_1);

  FullyConnectedOperator<float> fcOp(Fuseable::NoActivation<float>);
  fcOp
  .set_inputs({ 
    { FullyConnectedOperator<float>::input, in },
    { FullyConnectedOperator<float>::filter, w },
    { FullyConnectedOperator<float>::bias, b  }
  }).set_outputs({ 
    { FullyConnectedOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceFC, random_gen_fc__2) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<512*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 256,512 }, flt, s_ref_w_2);
  Tensor b  = new RomTensor({ 512 }, flt, s_ref_b_2);
  Tensor in = new RomTensor({ 1,256 }, flt, s_ref_in_2);
  Tensor out_ref = new RomTensor({ 1,512 }, flt, s_ref_out_2);
  Tensor out = new RamTensor({ 1,512 }, flt);

  FullyConnectedOperator<float> fcOp(Fuseable::NoActivation<float>);
  fcOp
  .set_inputs({ 
    { FullyConnectedOperator<float>::input, in },
    { FullyConnectedOperator<float>::filter, w },
    { FullyConnectedOperator<float>::bias, b  }
  }).set_outputs({ 
    { FullyConnectedOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceFC, random_gen_fc__3) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<512*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 512 }, flt, s_ref_b_3);
  Tensor out = new RamTensor({ 1,512 }, flt);
  Tensor out_ref = new RomTensor({ 1,512 }, flt, s_ref_out_3);
  Tensor in = new RomTensor({ 1,256 }, flt, s_ref_in_3);
  Tensor w = new RomTensor({ 256,512 }, flt, s_ref_w_3);

  FullyConnectedOperator<float> fcOp(Fuseable::NoActivation<float>);
  fcOp
  .set_inputs({ 
    { FullyConnectedOperator<float>::input, in },
    { FullyConnectedOperator<float>::filter, w },
    { FullyConnectedOperator<float>::bias, b  }
  }).set_outputs({ 
    { FullyConnectedOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceFC, random_gen_fc__4) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<512*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,512 }, flt);
  Tensor w = new RomTensor({ 256,512 }, flt, s_ref_w_4);
  Tensor in = new RomTensor({ 1,256 }, flt, s_ref_in_4);
  Tensor b  = new RomTensor({ 512 }, flt, s_ref_b_4);
  Tensor out_ref = new RomTensor({ 1,512 }, flt, s_ref_out_4);

  FullyConnectedOperator<float> fcOp(Fuseable::NoActivation<float>);
  fcOp
  .set_inputs({ 
    { FullyConnectedOperator<float>::input, in },
    { FullyConnectedOperator<float>::filter, w },
    { FullyConnectedOperator<float>::bias, b  }
  }).set_outputs({ 
    { FullyConnectedOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 0.0001);
}
}

