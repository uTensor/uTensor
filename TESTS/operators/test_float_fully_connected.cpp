#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "/Users/dboyliao/Work/open_source/uTensor/uTensor/TESTS/operators/constants_float_fully_connected.hpp"
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
  localCircularArenaAllocator<227*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,227 }, flt, s_ref_out_00);
  Tensor b  = new RomTensor({ 227 }, flt, s_ref_b_00);
  Tensor in = new RomTensor({ 1,146 }, flt, s_ref_in_00);
  Tensor w = new RomTensor({ 146,227 }, flt, s_ref_w_00);
  Tensor out = new RamTensor({ 1,227 }, flt);

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

  for(int i = 0; i < 227; i++) {
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
  localCircularArenaAllocator<439*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 439 }, flt, s_ref_b_01);
  Tensor in = new RomTensor({ 1,157 }, flt, s_ref_in_01);
  Tensor w = new RomTensor({ 157,439 }, flt, s_ref_w_01);
  Tensor out = new RamTensor({ 1,439 }, flt);
  Tensor out_ref = new RomTensor({ 1,439 }, flt, s_ref_out_01);

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

  for(int i = 0; i < 439; i++) {
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
  localCircularArenaAllocator<450*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 506,450 }, flt, s_ref_w_02);
  Tensor b  = new RomTensor({ 450 }, flt, s_ref_b_02);
  Tensor out_ref = new RomTensor({ 1,450 }, flt, s_ref_out_02);
  Tensor in = new RomTensor({ 1,506 }, flt, s_ref_in_02);
  Tensor out = new RamTensor({ 1,450 }, flt);

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

  for(int i = 0; i < 450; i++) {
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
  localCircularArenaAllocator<407*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 407 }, flt, s_ref_b_03);
  Tensor out = new RamTensor({ 1,407 }, flt);
  Tensor w = new RomTensor({ 510,407 }, flt, s_ref_w_03);
  Tensor out_ref = new RomTensor({ 1,407 }, flt, s_ref_out_03);
  Tensor in = new RomTensor({ 1,510 }, flt, s_ref_in_03);

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

  for(int i = 0; i < 407; i++) {
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
  localCircularArenaAllocator<376*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,376 }, flt, s_ref_out_04);
  Tensor in = new RomTensor({ 1,183 }, flt, s_ref_in_04);
  Tensor w = new RomTensor({ 183,376 }, flt, s_ref_w_04);
  Tensor out = new RamTensor({ 1,376 }, flt);
  Tensor b  = new RomTensor({ 376 }, flt, s_ref_b_04);

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

  for(int i = 0; i < 376; i++) {
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
  localCircularArenaAllocator<439*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,439 }, flt);
  Tensor b  = new RomTensor({ 439 }, flt, s_ref_b_05);
  Tensor in = new RomTensor({ 1,372 }, flt, s_ref_in_05);
  Tensor w = new RomTensor({ 372,439 }, flt, s_ref_w_05);
  Tensor out_ref = new RomTensor({ 1,439 }, flt, s_ref_out_05);

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

  for(int i = 0; i < 439; i++) {
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
  localCircularArenaAllocator<150*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,150 }, flt, s_ref_out_06);
  Tensor in = new RomTensor({ 1,270 }, flt, s_ref_in_06);
  Tensor out = new RamTensor({ 1,150 }, flt);
  Tensor b  = new RomTensor({ 150 }, flt, s_ref_b_06);
  Tensor w = new RomTensor({ 270,150 }, flt, s_ref_w_06);

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
 * Generated Test 8
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<457*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,457 }, flt);
  Tensor b  = new RomTensor({ 457 }, flt, s_ref_b_07);
  Tensor in = new RomTensor({ 1,280 }, flt, s_ref_in_07);
  Tensor out_ref = new RomTensor({ 1,457 }, flt, s_ref_out_07);
  Tensor w = new RomTensor({ 280,457 }, flt, s_ref_w_07);

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

  for(int i = 0; i < 457; i++) {
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
  localCircularArenaAllocator<497*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 497 }, flt, s_ref_b_08);
  Tensor out_ref = new RomTensor({ 1,497 }, flt, s_ref_out_08);
  Tensor out = new RamTensor({ 1,497 }, flt);
  Tensor in = new RomTensor({ 1,431 }, flt, s_ref_in_08);
  Tensor w = new RomTensor({ 431,497 }, flt, s_ref_w_08);

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

  for(int i = 0; i < 497; i++) {
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
  localCircularArenaAllocator<391*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,391 }, flt);
  Tensor out_ref = new RomTensor({ 1,391 }, flt, s_ref_out_09);
  Tensor b  = new RomTensor({ 391 }, flt, s_ref_b_09);
  Tensor w = new RomTensor({ 344,391 }, flt, s_ref_w_09);
  Tensor in = new RomTensor({ 1,344 }, flt, s_ref_in_09);

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

  for(int i = 0; i < 391; i++) {
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
  localCircularArenaAllocator<252*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,252 }, flt, s_ref_out_10);
  Tensor out = new RamTensor({ 1,252 }, flt);
  Tensor in = new RomTensor({ 1,260 }, flt, s_ref_in_10);
  Tensor b  = new RomTensor({ 252 }, flt, s_ref_b_10);
  Tensor w = new RomTensor({ 260,252 }, flt, s_ref_w_10);

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

  for(int i = 0; i < 252; i++) {
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
  localCircularArenaAllocator<289*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 384,289 }, flt, s_ref_w_11);
  Tensor out_ref = new RomTensor({ 1,289 }, flt, s_ref_out_11);
  Tensor b  = new RomTensor({ 289 }, flt, s_ref_b_11);
  Tensor out = new RamTensor({ 1,289 }, flt);
  Tensor in = new RomTensor({ 1,384 }, flt, s_ref_in_11);

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

  for(int i = 0; i < 289; i++) {
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
  localCircularArenaAllocator<286*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 393,286 }, flt, s_ref_w_12);
  Tensor out_ref = new RomTensor({ 1,286 }, flt, s_ref_out_12);
  Tensor out = new RamTensor({ 1,286 }, flt);
  Tensor in = new RomTensor({ 1,393 }, flt, s_ref_in_12);
  Tensor b  = new RomTensor({ 286 }, flt, s_ref_b_12);

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
 * Generated Test 14
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<146*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,386 }, flt, s_ref_in_13);
  Tensor b  = new RomTensor({ 146 }, flt, s_ref_b_13);
  Tensor out_ref = new RomTensor({ 1,146 }, flt, s_ref_out_13);
  Tensor w = new RomTensor({ 386,146 }, flt, s_ref_w_13);
  Tensor out = new RamTensor({ 1,146 }, flt);

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

  for(int i = 0; i < 146; i++) {
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
  localCircularArenaAllocator<474*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 474 }, flt, s_ref_b_14);
  Tensor out_ref = new RomTensor({ 1,474 }, flt, s_ref_out_14);
  Tensor in = new RomTensor({ 1,367 }, flt, s_ref_in_14);
  Tensor out = new RamTensor({ 1,474 }, flt);
  Tensor w = new RomTensor({ 367,474 }, flt, s_ref_w_14);

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

  for(int i = 0; i < 474; i++) {
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
  localCircularArenaAllocator<504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,504 }, flt);
  Tensor w = new RomTensor({ 193,504 }, flt, s_ref_w_15);
  Tensor b  = new RomTensor({ 504 }, flt, s_ref_b_15);
  Tensor out_ref = new RomTensor({ 1,504 }, flt, s_ref_out_15);
  Tensor in = new RomTensor({ 1,193 }, flt, s_ref_in_15);

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
 * Generated Test 17
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<304*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,304 }, flt, s_ref_out_16);
  Tensor out = new RamTensor({ 1,304 }, flt);
  Tensor b  = new RomTensor({ 304 }, flt, s_ref_b_16);
  Tensor w = new RomTensor({ 478,304 }, flt, s_ref_w_16);
  Tensor in = new RomTensor({ 1,478 }, flt, s_ref_in_16);

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

  for(int i = 0; i < 304; i++) {
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
  localCircularArenaAllocator<410*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 410 }, flt, s_ref_b_17);
  Tensor out_ref = new RomTensor({ 1,410 }, flt, s_ref_out_17);
  Tensor in = new RomTensor({ 1,350 }, flt, s_ref_in_17);
  Tensor out = new RamTensor({ 1,410 }, flt);
  Tensor w = new RomTensor({ 350,410 }, flt, s_ref_w_17);

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

  for(int i = 0; i < 410; i++) {
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
  localCircularArenaAllocator<322*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 215,322 }, flt, s_ref_w_18);
  Tensor in = new RomTensor({ 1,215 }, flt, s_ref_in_18);
  Tensor b  = new RomTensor({ 322 }, flt, s_ref_b_18);
  Tensor out = new RamTensor({ 1,322 }, flt);
  Tensor out_ref = new RomTensor({ 1,322 }, flt, s_ref_out_18);

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

  for(int i = 0; i < 322; i++) {
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
  localCircularArenaAllocator<229*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,308 }, flt, s_ref_in_19);
  Tensor out_ref = new RomTensor({ 1,229 }, flt, s_ref_out_19);
  Tensor b  = new RomTensor({ 229 }, flt, s_ref_b_19);
  Tensor w = new RomTensor({ 308,229 }, flt, s_ref_w_19);
  Tensor out = new RamTensor({ 1,229 }, flt);

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

  for(int i = 0; i < 229; i++) {
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
  localCircularArenaAllocator<268*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor w = new RomTensor({ 181,268 }, flt, s_ref_w_20);
  Tensor b  = new RomTensor({ 268 }, flt, s_ref_b_20);
  Tensor in = new RomTensor({ 1,181 }, flt, s_ref_in_20);
  Tensor out = new RamTensor({ 1,268 }, flt);
  Tensor out_ref = new RomTensor({ 1,268 }, flt, s_ref_out_20);

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

  for(int i = 0; i < 268; i++) {
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
  localCircularArenaAllocator<351*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,351 }, flt, s_ref_out_21);
  Tensor w = new RomTensor({ 358,351 }, flt, s_ref_w_21);
  Tensor in = new RomTensor({ 1,358 }, flt, s_ref_in_21);
  Tensor b  = new RomTensor({ 351 }, flt, s_ref_b_21);
  Tensor out = new RamTensor({ 1,351 }, flt);

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
 * Generated Test 23
 ***************************************/
TEST(ReferenceFloatFullyConnect, random_gen_fc__22) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<456*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,456 }, flt);
  Tensor out_ref = new RomTensor({ 1,456 }, flt, s_ref_out_22);
  Tensor in = new RomTensor({ 1,448 }, flt, s_ref_in_22);
  Tensor w = new RomTensor({ 448,456 }, flt, s_ref_w_22);
  Tensor b  = new RomTensor({ 456 }, flt, s_ref_b_22);

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

  for(int i = 0; i < 456; i++) {
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
  localCircularArenaAllocator<187*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,187 }, flt, s_ref_out_23);
  Tensor b  = new RomTensor({ 187 }, flt, s_ref_b_23);
  Tensor out = new RamTensor({ 1,187 }, flt);
  Tensor in = new RomTensor({ 1,282 }, flt, s_ref_in_23);
  Tensor w = new RomTensor({ 282,187 }, flt, s_ref_w_23);

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

  for(int i = 0; i < 187; i++) {
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
  localCircularArenaAllocator<507*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,507 }, flt, s_ref_out_24);
  Tensor out = new RamTensor({ 1,507 }, flt);
  Tensor b  = new RomTensor({ 507 }, flt, s_ref_b_24);
  Tensor in = new RomTensor({ 1,203 }, flt, s_ref_in_24);
  Tensor w = new RomTensor({ 203,507 }, flt, s_ref_w_24);

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

  for(int i = 0; i < 507; i++) {
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
  localCircularArenaAllocator<202*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,202 }, flt);
  Tensor out_ref = new RomTensor({ 1,202 }, flt, s_ref_out_25);
  Tensor b  = new RomTensor({ 202 }, flt, s_ref_b_25);
  Tensor in = new RomTensor({ 1,395 }, flt, s_ref_in_25);
  Tensor w = new RomTensor({ 395,202 }, flt, s_ref_w_25);

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

  for(int i = 0; i < 202; i++) {
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
  localCircularArenaAllocator<184*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b  = new RomTensor({ 184 }, flt, s_ref_b_26);
  Tensor out_ref = new RomTensor({ 1,184 }, flt, s_ref_out_26);
  Tensor w = new RomTensor({ 399,184 }, flt, s_ref_w_26);
  Tensor in = new RomTensor({ 1,399 }, flt, s_ref_in_26);
  Tensor out = new RamTensor({ 1,184 }, flt);

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

  for(int i = 0; i < 184; i++) {
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
  localCircularArenaAllocator<221*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,372 }, flt, s_ref_in_27);
  Tensor b  = new RomTensor({ 221 }, flt, s_ref_b_27);
  Tensor w = new RomTensor({ 372,221 }, flt, s_ref_w_27);
  Tensor out = new RamTensor({ 1,221 }, flt);
  Tensor out_ref = new RomTensor({ 1,221 }, flt, s_ref_out_27);

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

  for(int i = 0; i < 221; i++) {
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
  localCircularArenaAllocator<490*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,490 }, flt);
  Tensor b  = new RomTensor({ 490 }, flt, s_ref_b_28);
  Tensor in = new RomTensor({ 1,463 }, flt, s_ref_in_28);
  Tensor out_ref = new RomTensor({ 1,490 }, flt, s_ref_out_28);
  Tensor w = new RomTensor({ 463,490 }, flt, s_ref_w_28);

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

  for(int i = 0; i < 490; i++) {
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
  localCircularArenaAllocator<151*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,151 }, flt, s_ref_out_29);
  Tensor w = new RomTensor({ 389,151 }, flt, s_ref_w_29);
  Tensor out = new RamTensor({ 1,151 }, flt);
  Tensor in = new RomTensor({ 1,389 }, flt, s_ref_in_29);
  Tensor b  = new RomTensor({ 151 }, flt, s_ref_b_29);

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

  for(int i = 0; i < 151; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

