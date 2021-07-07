#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_float_tanh.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test 1
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<14*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,14 }, flt);
  Tensor out_ref = new RomTensor({ 1,14 }, flt, s_ref_out_00);
  Tensor in = new RomTensor({ 1,14 }, flt, s_ref_in_00);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 14; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__01) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<40*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,40 }, flt, s_ref_in_01);
  Tensor out_ref = new RomTensor({ 1,40 }, flt, s_ref_out_01);
  Tensor out = new RamTensor({ 1,40 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 40; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<20*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,20 }, flt, s_ref_in_02);
  Tensor out_ref = new RomTensor({ 1,20 }, flt, s_ref_out_02);
  Tensor out = new RamTensor({ 1,20 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 20; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<22*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,22 }, flt);
  Tensor out_ref = new RomTensor({ 1,22 }, flt, s_ref_out_03);
  Tensor in = new RomTensor({ 1,22 }, flt, s_ref_in_03);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 22; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<11*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,11 }, flt, s_ref_in_04);
  Tensor out = new RamTensor({ 1,11 }, flt);
  Tensor out_ref = new RomTensor({ 1,11 }, flt, s_ref_out_04);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 11; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__05) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<16*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,16 }, flt, s_ref_out_05);
  Tensor out = new RamTensor({ 1,16 }, flt);
  Tensor in = new RomTensor({ 1,16 }, flt, s_ref_in_05);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 16; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 7
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<33*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,33 }, flt, s_ref_out_06);
  Tensor in = new RomTensor({ 1,33 }, flt, s_ref_in_06);
  Tensor out = new RamTensor({ 1,33 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 33; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 8
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__07) {
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

  Tensor out_ref = new RomTensor({ 1,30 }, flt, s_ref_out_07);
  Tensor out = new RamTensor({ 1,30 }, flt);
  Tensor in = new RomTensor({ 1,30 }, flt, s_ref_in_07);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 30; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 9
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__08) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<42*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,42 }, flt, s_ref_in_08);
  Tensor out_ref = new RomTensor({ 1,42 }, flt, s_ref_out_08);
  Tensor out = new RamTensor({ 1,42 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 42; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 10
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<10*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out_ref = new RomTensor({ 1,10 }, flt, s_ref_out_09);
  Tensor out = new RamTensor({ 1,10 }, flt);
  Tensor in = new RomTensor({ 1,10 }, flt, s_ref_in_09);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 10; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 11
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__10) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<47*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,47 }, flt, s_ref_in_10);
  Tensor out_ref = new RomTensor({ 1,47 }, flt, s_ref_out_10);
  Tensor out = new RamTensor({ 1,47 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 47; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 12
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__11) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<34*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,34 }, flt);
  Tensor in = new RomTensor({ 1,34 }, flt, s_ref_in_11);
  Tensor out_ref = new RomTensor({ 1,34 }, flt, s_ref_out_11);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 34; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 13
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<16*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,16 }, flt, s_ref_in_12);
  Tensor out = new RamTensor({ 1,16 }, flt);
  Tensor out_ref = new RomTensor({ 1,16 }, flt, s_ref_out_12);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 16; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 14
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<15*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,15 }, flt, s_ref_in_13);
  Tensor out_ref = new RomTensor({ 1,15 }, flt, s_ref_out_13);
  Tensor out = new RamTensor({ 1,15 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 15; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 15
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<43*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,43 }, flt, s_ref_in_14);
  Tensor out_ref = new RomTensor({ 1,43 }, flt, s_ref_out_14);
  Tensor out = new RamTensor({ 1,43 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 43; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 16
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<40*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,40 }, flt, s_ref_in_15);
  Tensor out = new RamTensor({ 1,40 }, flt);
  Tensor out_ref = new RomTensor({ 1,40 }, flt, s_ref_out_15);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 40; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 17
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<46*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,46 }, flt, s_ref_in_16);
  Tensor out_ref = new RomTensor({ 1,46 }, flt, s_ref_out_16);
  Tensor out = new RamTensor({ 1,46 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 46; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 18
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__17) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<24*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,24 }, flt, s_ref_in_17);
  Tensor out = new RamTensor({ 1,24 }, flt);
  Tensor out_ref = new RomTensor({ 1,24 }, flt, s_ref_out_17);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 24; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 19
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<38*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,38 }, flt, s_ref_in_18);
  Tensor out = new RamTensor({ 1,38 }, flt);
  Tensor out_ref = new RomTensor({ 1,38 }, flt, s_ref_out_18);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 38; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 20
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__19) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<39*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,39 }, flt);
  Tensor out_ref = new RomTensor({ 1,39 }, flt, s_ref_out_19);
  Tensor in = new RomTensor({ 1,39 }, flt, s_ref_in_19);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 39; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 21
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__20) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<14*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,14 }, flt);
  Tensor in = new RomTensor({ 1,14 }, flt, s_ref_in_20);
  Tensor out_ref = new RomTensor({ 1,14 }, flt, s_ref_out_20);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 14; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 22
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__21) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<22*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,22 }, flt, s_ref_in_21);
  Tensor out_ref = new RomTensor({ 1,22 }, flt, s_ref_out_21);
  Tensor out = new RamTensor({ 1,22 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 22; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 23
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__22) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<27*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,27 }, flt);
  Tensor out_ref = new RomTensor({ 1,27 }, flt, s_ref_out_22);
  Tensor in = new RomTensor({ 1,27 }, flt, s_ref_in_22);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 27; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 24
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__23) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<22*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,22 }, flt);
  Tensor out_ref = new RomTensor({ 1,22 }, flt, s_ref_out_23);
  Tensor in = new RomTensor({ 1,22 }, flt, s_ref_in_23);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 22; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 25
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__24) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<40*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,40 }, flt, s_ref_in_24);
  Tensor out = new RamTensor({ 1,40 }, flt);
  Tensor out_ref = new RomTensor({ 1,40 }, flt, s_ref_out_24);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 40; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 26
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__25) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<27*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,27 }, flt, s_ref_in_25);
  Tensor out = new RamTensor({ 1,27 }, flt);
  Tensor out_ref = new RomTensor({ 1,27 }, flt, s_ref_out_25);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 27; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 27
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__26) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<44*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,44 }, flt, s_ref_in_26);
  Tensor out_ref = new RomTensor({ 1,44 }, flt, s_ref_out_26);
  Tensor out = new RamTensor({ 1,44 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 44; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 28
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__27) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<50*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,50 }, flt, s_ref_in_27);
  Tensor out_ref = new RomTensor({ 1,50 }, flt, s_ref_out_27);
  Tensor out = new RamTensor({ 1,50 }, flt);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 50; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 29
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__28) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<37*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,37 }, flt);
  Tensor in = new RomTensor({ 1,37 }, flt, s_ref_in_28);
  Tensor out_ref = new RomTensor({ 1,37 }, flt, s_ref_out_28);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 37; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

/***************************************
 * Generated Test 30
 ***************************************/
TEST(ReferenceFloatTanh, random_gen_float_tanh__29) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in = new RomTensor({ 1,18 }, flt, s_ref_in_29);
  Tensor out = new RamTensor({ 1,18 }, flt);
  Tensor out_ref = new RomTensor({ 1,18 }, flt, s_ref_out_29);

  ReferenceOperators::TanhOperator<float,float> tanh_op;
  tanh_op
  .set_inputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_in, in }
  }).set_outputs({ 
    { ReferenceOperators::TanhOperator<float,float>::act_out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 18; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ), 1e-05);
}

}

