#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_float_fully_connected.hpp"
using std::cout;
using std::endl;

using namespace uTensor;
using namespace uTensor::ReferenceOperators;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test 1
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<259*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 259 }, flt, s_ref_b_00);
  Tensor out = new RamTensor({ 1,259 }, flt);
  Tensor in = new RomTensor({ 1,326 }, flt, s_ref_in_00);
  Tensor out_ref = new RomTensor({ 1,259 }, flt, s_ref_out_00);
  Tensor w = new RomTensor({ 326,259 }, flt, s_ref_w_00);

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

  for(int i = 0; i < 259; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__01) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<310*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,329 }, flt, s_ref_in_01);
  Tensor out_ref = new RomTensor({ 1,310 }, flt, s_ref_out_01);
  Tensor b  = new RomTensor({ 310 }, flt, s_ref_b_01);
  Tensor out = new RamTensor({ 1,310 }, flt);
  Tensor w = new RomTensor({ 329,310 }, flt, s_ref_w_01);

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

  for(int i = 0; i < 310; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<286*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,286 }, flt, s_ref_out_02);
  Tensor out = new RamTensor({ 1,286 }, flt);
  Tensor in = new RomTensor({ 1,239 }, flt, s_ref_in_02);
  Tensor w = new RomTensor({ 239,286 }, flt, s_ref_w_02);
  Tensor b  = new RomTensor({ 286 }, flt, s_ref_b_02);

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

  for(int i = 0; i < 286; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<467*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 309,467 }, flt, s_ref_w_03);
  Tensor in = new RomTensor({ 1,309 }, flt, s_ref_in_03);
  Tensor out = new RamTensor({ 1,467 }, flt);
  Tensor b  = new RomTensor({ 467 }, flt, s_ref_b_03);
  Tensor out_ref = new RomTensor({ 1,467 }, flt, s_ref_out_03);

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

  for(int i = 0; i < 467; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<429*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 276,429 }, flt, s_ref_w_04);
  Tensor out = new RamTensor({ 1,429 }, flt);
  Tensor b  = new RomTensor({ 429 }, flt, s_ref_b_04);
  Tensor out_ref = new RomTensor({ 1,429 }, flt, s_ref_out_04);
  Tensor in = new RomTensor({ 1,276 }, flt, s_ref_in_04);

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

  for(int i = 0; i < 429; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__05) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<169*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 163,169 }, flt, s_ref_w_05);
  Tensor out_ref = new RomTensor({ 1,169 }, flt, s_ref_out_05);
  Tensor in = new RomTensor({ 1,163 }, flt, s_ref_in_05);
  Tensor out = new RamTensor({ 1,169 }, flt);
  Tensor b  = new RomTensor({ 169 }, flt, s_ref_b_05);

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

  for(int i = 0; i < 169; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 7
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<295*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,295 }, flt);
  Tensor b  = new RomTensor({ 295 }, flt, s_ref_b_06);
  Tensor in = new RomTensor({ 1,377 }, flt, s_ref_in_06);
  Tensor out_ref = new RomTensor({ 1,295 }, flt, s_ref_out_06);
  Tensor w = new RomTensor({ 377,295 }, flt, s_ref_w_06);

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

  for(int i = 0; i < 295; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 8
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<143*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 503,143 }, flt, s_ref_w_07);
  Tensor in = new RomTensor({ 1,503 }, flt, s_ref_in_07);
  Tensor out = new RamTensor({ 1,143 }, flt);
  Tensor b  = new RomTensor({ 143 }, flt, s_ref_b_07);
  Tensor out_ref = new RomTensor({ 1,143 }, flt, s_ref_out_07);

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

  for(int i = 0; i < 143; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 9
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__08) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<135*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,135 }, flt, s_ref_out_08);
  Tensor in = new RomTensor({ 1,495 }, flt, s_ref_in_08);
  Tensor w = new RomTensor({ 495,135 }, flt, s_ref_w_08);
  Tensor b  = new RomTensor({ 135 }, flt, s_ref_b_08);
  Tensor out = new RamTensor({ 1,135 }, flt);

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

  for(int i = 0; i < 135; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 10
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<338*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 338 }, flt, s_ref_b_09);
  Tensor out_ref = new RomTensor({ 1,338 }, flt, s_ref_out_09);
  Tensor w = new RomTensor({ 503,338 }, flt, s_ref_w_09);
  Tensor in = new RomTensor({ 1,503 }, flt, s_ref_in_09);
  Tensor out = new RamTensor({ 1,338 }, flt);

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

  for(int i = 0; i < 338; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 11
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__10) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<249*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,249 }, flt);
  Tensor b  = new RomTensor({ 249 }, flt, s_ref_b_10);
  Tensor out_ref = new RomTensor({ 1,249 }, flt, s_ref_out_10);
  Tensor w = new RomTensor({ 492,249 }, flt, s_ref_w_10);
  Tensor in = new RomTensor({ 1,492 }, flt, s_ref_in_10);

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

  for(int i = 0; i < 249; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 12
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__11) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<357*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,357 }, flt, s_ref_out_11);
  Tensor w = new RomTensor({ 206,357 }, flt, s_ref_w_11);
  Tensor in = new RomTensor({ 1,206 }, flt, s_ref_in_11);
  Tensor out = new RamTensor({ 1,357 }, flt);
  Tensor b  = new RomTensor({ 357 }, flt, s_ref_b_11);

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

  for(int i = 0; i < 357; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 13
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<403*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 403 }, flt, s_ref_b_12);
  Tensor w = new RomTensor({ 178,403 }, flt, s_ref_w_12);
  Tensor in = new RomTensor({ 1,178 }, flt, s_ref_in_12);
  Tensor out_ref = new RomTensor({ 1,403 }, flt, s_ref_out_12);
  Tensor out = new RamTensor({ 1,403 }, flt);

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

  for(int i = 0; i < 403; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 14
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<150*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,150 }, flt);
  Tensor out_ref = new RomTensor({ 1,150 }, flt, s_ref_out_13);
  Tensor in = new RomTensor({ 1,388 }, flt, s_ref_in_13);
  Tensor w = new RomTensor({ 388,150 }, flt, s_ref_w_13);
  Tensor b  = new RomTensor({ 150 }, flt, s_ref_b_13);

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

  for(int i = 0; i < 150; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 15
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<423*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 423 }, flt, s_ref_b_14);
  Tensor out = new RamTensor({ 1,423 }, flt);
  Tensor w = new RomTensor({ 387,423 }, flt, s_ref_w_14);
  Tensor out_ref = new RomTensor({ 1,423 }, flt, s_ref_out_14);
  Tensor in = new RomTensor({ 1,387 }, flt, s_ref_in_14);

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

  for(int i = 0; i < 423; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 16
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<300*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,300 }, flt);
  Tensor in = new RomTensor({ 1,454 }, flt, s_ref_in_15);
  Tensor out_ref = new RomTensor({ 1,300 }, flt, s_ref_out_15);
  Tensor w = new RomTensor({ 454,300 }, flt, s_ref_w_15);
  Tensor b  = new RomTensor({ 300 }, flt, s_ref_b_15);

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

  for(int i = 0; i < 300; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 17
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,504 }, flt);
  Tensor out_ref = new RomTensor({ 1,504 }, flt, s_ref_out_16);
  Tensor w = new RomTensor({ 130,504 }, flt, s_ref_w_16);
  Tensor b  = new RomTensor({ 504 }, flt, s_ref_b_16);
  Tensor in = new RomTensor({ 1,130 }, flt, s_ref_in_16);

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

  for(int i = 0; i < 504; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 18
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__17) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<138*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,138 }, flt, s_ref_out_17);
  Tensor w = new RomTensor({ 278,138 }, flt, s_ref_w_17);
  Tensor b  = new RomTensor({ 138 }, flt, s_ref_b_17);
  Tensor out = new RamTensor({ 1,138 }, flt);
  Tensor in = new RomTensor({ 1,278 }, flt, s_ref_in_17);

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

  for(int i = 0; i < 138; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 19
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<290*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 372,290 }, flt, s_ref_w_18);
  Tensor in = new RomTensor({ 1,372 }, flt, s_ref_in_18);
  Tensor out_ref = new RomTensor({ 1,290 }, flt, s_ref_out_18);
  Tensor out = new RamTensor({ 1,290 }, flt);
  Tensor b  = new RomTensor({ 290 }, flt, s_ref_b_18);

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

  for(int i = 0; i < 290; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 20
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__19) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<216*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,216 }, flt, s_ref_out_19);
  Tensor out = new RamTensor({ 1,216 }, flt);
  Tensor b  = new RomTensor({ 216 }, flt, s_ref_b_19);
  Tensor in = new RomTensor({ 1,471 }, flt, s_ref_in_19);
  Tensor w = new RomTensor({ 471,216 }, flt, s_ref_w_19);

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

  for(int i = 0; i < 216; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 21
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__20) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<178*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 178 }, flt, s_ref_b_20);
  Tensor out = new RamTensor({ 1,178 }, flt);
  Tensor w = new RomTensor({ 509,178 }, flt, s_ref_w_20);
  Tensor out_ref = new RomTensor({ 1,178 }, flt, s_ref_out_20);
  Tensor in = new RomTensor({ 1,509 }, flt, s_ref_in_20);

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

  for(int i = 0; i < 178; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 22
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__21) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<316*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,316 }, flt);
  Tensor in = new RomTensor({ 1,403 }, flt, s_ref_in_21);
  Tensor b  = new RomTensor({ 316 }, flt, s_ref_b_21);
  Tensor out_ref = new RomTensor({ 1,316 }, flt, s_ref_out_21);
  Tensor w = new RomTensor({ 403,316 }, flt, s_ref_w_21);

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

  for(int i = 0; i < 316; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 23
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__22) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<295*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,295 }, flt, s_ref_out_22);
  Tensor b  = new RomTensor({ 295 }, flt, s_ref_b_22);
  Tensor in = new RomTensor({ 1,325 }, flt, s_ref_in_22);
  Tensor out = new RamTensor({ 1,295 }, flt);
  Tensor w = new RomTensor({ 325,295 }, flt, s_ref_w_22);

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

  for(int i = 0; i < 295; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 24
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__23) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<140*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,140 }, flt);
  Tensor in = new RomTensor({ 1,421 }, flt, s_ref_in_23);
  Tensor w = new RomTensor({ 421,140 }, flt, s_ref_w_23);
  Tensor b  = new RomTensor({ 140 }, flt, s_ref_b_23);
  Tensor out_ref = new RomTensor({ 1,140 }, flt, s_ref_out_23);

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

  for(int i = 0; i < 140; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 25
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__24) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<351*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 351 }, flt, s_ref_b_24);
  Tensor w = new RomTensor({ 215,351 }, flt, s_ref_w_24);
  Tensor out = new RamTensor({ 1,351 }, flt);
  Tensor out_ref = new RomTensor({ 1,351 }, flt, s_ref_out_24);
  Tensor in = new RomTensor({ 1,215 }, flt, s_ref_in_24);

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

  for(int i = 0; i < 351; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 26
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__25) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<177*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,267 }, flt, s_ref_in_25);
  Tensor b  = new RomTensor({ 177 }, flt, s_ref_b_25);
  Tensor out = new RamTensor({ 1,177 }, flt);
  Tensor out_ref = new RomTensor({ 1,177 }, flt, s_ref_out_25);
  Tensor w = new RomTensor({ 267,177 }, flt, s_ref_w_25);

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

  for(int i = 0; i < 177; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 27
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__26) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<348*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,348 }, flt);
  Tensor w = new RomTensor({ 456,348 }, flt, s_ref_w_26);
  Tensor out_ref = new RomTensor({ 1,348 }, flt, s_ref_out_26);
  Tensor b  = new RomTensor({ 348 }, flt, s_ref_b_26);
  Tensor in = new RomTensor({ 1,456 }, flt, s_ref_in_26);

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

  for(int i = 0; i < 348; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 28
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__27) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<217*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,217 }, flt);
  Tensor b  = new RomTensor({ 217 }, flt, s_ref_b_27);
  Tensor in = new RomTensor({ 1,495 }, flt, s_ref_in_27);
  Tensor w = new RomTensor({ 495,217 }, flt, s_ref_w_27);
  Tensor out_ref = new RomTensor({ 1,217 }, flt, s_ref_out_27);

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

  for(int i = 0; i < 217; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 29
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__28) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<508*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 415,508 }, flt, s_ref_w_28);
  Tensor out = new RamTensor({ 1,508 }, flt);
  Tensor b  = new RomTensor({ 508 }, flt, s_ref_b_28);
  Tensor in = new RomTensor({ 1,415 }, flt, s_ref_in_28);
  Tensor out_ref = new RomTensor({ 1,508 }, flt, s_ref_out_28);

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

  for(int i = 0; i < 508; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 30
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__29) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<418*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,192 }, flt, s_ref_in_29);
  Tensor b  = new RomTensor({ 418 }, flt, s_ref_b_29);
  Tensor w = new RomTensor({ 192,418 }, flt, s_ref_w_29);
  Tensor out_ref = new RomTensor({ 1,418 }, flt, s_ref_out_29);
  Tensor out = new RamTensor({ 1,418 }, flt);

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

  for(int i = 0; i < 418; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

