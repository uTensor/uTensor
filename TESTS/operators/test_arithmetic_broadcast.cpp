#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_arithmetic_broadcast.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test 1
 ***************************************/
TEST(ReferenceSub, random_gen_sub__00) {
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

  Tensor out = new RamTensor({ 20,3,5 }, flt);
  Tensor in1 = new RomTensor({ 20,3,5 }, flt, s_ref_sub_in1_00);
  Tensor ref_out = new RomTensor({ 20,3,5 }, flt, s_ref_sub_out_00);
  Tensor in2 = new RomTensor({ 5 }, flt, s_ref_sub_in2_00);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 300; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceAdd, random_gen_add__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<250*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 25,1 }, flt, s_ref_add_in2_00);
  Tensor out = new RamTensor({ 10,25,1 }, flt);
  Tensor in1 = new RomTensor({ 10,25,1 }, flt, s_ref_add_in1_00);
  Tensor ref_out = new RomTensor({ 10,25,1 }, flt, s_ref_add_out_00);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 250; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceMul, random_gen_mul__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<288*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,4 }, flt, s_ref_mul_in2_00);
  Tensor out = new RamTensor({ 9,8,1,4 }, flt);
  Tensor ref_out = new RomTensor({ 9,8,1,4 }, flt, s_ref_mul_out_00);
  Tensor in1 = new RomTensor({ 9,8,1,4 }, flt, s_ref_mul_in1_00);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 288; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceDiv, random_gen_div__00) {
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

  Tensor ref_out = new RomTensor({ 21,1,1 }, flt, s_ref_div_out_00);
  Tensor in2 = new RomTensor({ 1,1 }, flt, s_ref_div_in2_00);
  Tensor in1 = new RomTensor({ 21,1,1 }, flt, s_ref_div_in1_00);
  Tensor out = new RamTensor({ 21,1,1 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 21; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceSub, random_gen_sub__01) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<95*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 5,1 }, flt, s_ref_sub_in2_01);
  Tensor out = new RamTensor({ 19,5,1 }, flt);
  Tensor ref_out = new RomTensor({ 19,5,1 }, flt, s_ref_sub_out_01);
  Tensor in1 = new RomTensor({ 19,5,1 }, flt, s_ref_sub_in1_01);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 95; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceAdd, random_gen_add__01) {
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

  Tensor out = new RamTensor({ 12,25,1 }, flt);
  Tensor in2 = new RomTensor({ 25,1 }, flt, s_ref_add_in2_01);
  Tensor in1 = new RomTensor({ 12,25,1 }, flt, s_ref_add_in1_01);
  Tensor ref_out = new RomTensor({ 12,25,1 }, flt, s_ref_add_out_01);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 300; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 7
 ***************************************/
TEST(ReferenceMul, random_gen_mul__01) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<378*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 6,1,9 }, flt, s_ref_mul_in2_01);
  Tensor in1 = new RomTensor({ 7,6,1,9 }, flt, s_ref_mul_in1_01);
  Tensor ref_out = new RomTensor({ 7,6,1,9 }, flt, s_ref_mul_out_01);
  Tensor out = new RamTensor({ 7,6,1,9 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 378; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 8
 ***************************************/
TEST(ReferenceDiv, random_gen_div__01) {
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

  Tensor in1 = new RomTensor({ 21,1,24,1 }, flt, s_ref_div_in1_01);
  Tensor out = new RamTensor({ 21,1,24,1 }, flt);
  Tensor ref_out = new RomTensor({ 21,1,24,1 }, flt, s_ref_div_out_01);
  Tensor in2 = new RomTensor({ 24,1 }, flt, s_ref_div_in2_01);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 504; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 9
 ***************************************/
TEST(ReferenceSub, random_gen_sub__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1463*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 11,19,7,1 }, flt, s_ref_sub_in1_02);
  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_sub_in2_02);
  Tensor out = new RamTensor({ 11,19,7,1 }, flt);
  Tensor ref_out = new RomTensor({ 11,19,7,1 }, flt, s_ref_sub_out_02);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1463; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 10
 ***************************************/
TEST(ReferenceAdd, random_gen_add__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<8694*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 23,1,18,21 }, flt, s_ref_add_out_02);
  Tensor in1 = new RomTensor({ 23,1,18,21 }, flt, s_ref_add_in1_02);
  Tensor out = new RamTensor({ 23,1,18,21 }, flt);
  Tensor in2 = new RomTensor({ 1,18,21 }, flt, s_ref_add_in2_02);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 8694; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 11
 ***************************************/
TEST(ReferenceMul, random_gen_mul__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2700*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 18,6,25 }, flt, s_ref_mul_in1_02);
  Tensor out = new RamTensor({ 18,6,25 }, flt);
  Tensor in2 = new RomTensor({ 25 }, flt, s_ref_mul_in2_02);
  Tensor ref_out = new RomTensor({ 18,6,25 }, flt, s_ref_mul_out_02);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 2700; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 12
 ***************************************/
TEST(ReferenceDiv, random_gen_div__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<560*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 8,14,5,1 }, flt);
  Tensor in2 = new RomTensor({ 5,1 }, flt, s_ref_div_in2_02);
  Tensor in1 = new RomTensor({ 8,14,5,1 }, flt, s_ref_div_in1_02);
  Tensor ref_out = new RomTensor({ 8,14,5,1 }, flt, s_ref_div_out_02);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 560; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 13
 ***************************************/
TEST(ReferenceSub, random_gen_sub__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<54*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 9,1,1,6 }, flt, s_ref_sub_in1_03);
  Tensor ref_out = new RomTensor({ 9,1,1,6 }, flt, s_ref_sub_out_03);
  Tensor in2 = new RomTensor({ 1,1,6 }, flt, s_ref_sub_in2_03);
  Tensor out = new RamTensor({ 9,1,1,6 }, flt);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 54; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 14
 ***************************************/
TEST(ReferenceAdd, random_gen_add__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 14,12,1 }, flt, s_ref_add_in1_03);
  Tensor ref_out = new RomTensor({ 14,12,1 }, flt, s_ref_add_out_03);
  Tensor in2 = new RomTensor({ 12,1 }, flt, s_ref_add_in2_03);
  Tensor out = new RamTensor({ 14,12,1 }, flt);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 168; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 15
 ***************************************/
TEST(ReferenceMul, random_gen_mul__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5625*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 15,25,15 }, flt, s_ref_mul_out_03);
  Tensor in1 = new RomTensor({ 15,25,15 }, flt, s_ref_mul_in1_03);
  Tensor out = new RamTensor({ 15,25,15 }, flt);
  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_mul_in2_03);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5625; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 16
 ***************************************/
TEST(ReferenceDiv, random_gen_div__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1440*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,18,4 }, flt, s_ref_div_in2_03);
  Tensor out = new RamTensor({ 20,1,18,4 }, flt);
  Tensor ref_out = new RomTensor({ 20,1,18,4 }, flt, s_ref_div_out_03);
  Tensor in1 = new RomTensor({ 20,1,18,4 }, flt, s_ref_div_in1_03);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1440; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 17
 ***************************************/
TEST(ReferenceSub, random_gen_sub__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<126*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,21,1 }, flt, s_ref_sub_in2_04);
  Tensor out = new RamTensor({ 6,1,21,1 }, flt);
  Tensor in1 = new RomTensor({ 6,1,21,1 }, flt, s_ref_sub_in1_04);
  Tensor ref_out = new RomTensor({ 6,1,21,1 }, flt, s_ref_sub_out_04);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 126; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 18
 ***************************************/
TEST(ReferenceAdd, random_gen_add__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<378*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 18,21,1 }, flt, s_ref_add_out_04);
  Tensor in2 = new RomTensor({ 21,1 }, flt, s_ref_add_in2_04);
  Tensor in1 = new RomTensor({ 18,21,1 }, flt, s_ref_add_in1_04);
  Tensor out = new RamTensor({ 18,21,1 }, flt);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 378; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 19
 ***************************************/
TEST(ReferenceMul, random_gen_mul__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1008*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,3 }, flt, s_ref_mul_in2_04);
  Tensor ref_out = new RomTensor({ 16,21,1,3 }, flt, s_ref_mul_out_04);
  Tensor in1 = new RomTensor({ 16,21,1,3 }, flt, s_ref_mul_in1_04);
  Tensor out = new RamTensor({ 16,21,1,3 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1008; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 20
 ***************************************/
TEST(ReferenceDiv, random_gen_div__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1190*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 5,2,7,17 }, flt);
  Tensor ref_out = new RomTensor({ 5,2,7,17 }, flt, s_ref_div_out_04);
  Tensor in1 = new RomTensor({ 5,2,7,17 }, flt, s_ref_div_in1_04);
  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_div_in2_04);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1190; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 21
 ***************************************/
TEST(ReferenceSub, random_gen_sub__05) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<54*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 1,6,1,9 }, flt, s_ref_sub_in1_05);
  Tensor out = new RamTensor({ 1,6,1,9 }, flt);
  Tensor ref_out = new RomTensor({ 1,6,1,9 }, flt, s_ref_sub_out_05);
  Tensor in2 = new RomTensor({ 1,9 }, flt, s_ref_sub_in2_05);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 54; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 22
 ***************************************/
TEST(ReferenceAdd, random_gen_add__05) {
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

  Tensor in2 = new RomTensor({ 2 }, flt, s_ref_add_in2_05);
  Tensor ref_out = new RomTensor({ 8,8,4,2 }, flt, s_ref_add_out_05);
  Tensor out = new RamTensor({ 8,8,4,2 }, flt);
  Tensor in1 = new RomTensor({ 8,8,4,2 }, flt, s_ref_add_in1_05);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 512; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 23
 ***************************************/
TEST(ReferenceMul, random_gen_mul__05) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<7056*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 21,16,21 }, flt, s_ref_mul_in1_05);
  Tensor out = new RamTensor({ 21,16,21 }, flt);
  Tensor ref_out = new RomTensor({ 21,16,21 }, flt, s_ref_mul_out_05);
  Tensor in2 = new RomTensor({ 21 }, flt, s_ref_mul_in2_05);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 7056; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 24
 ***************************************/
TEST(ReferenceDiv, random_gen_div__05) {
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

  Tensor in1 = new RomTensor({ 3,8,1 }, flt, s_ref_div_in1_05);
  Tensor out = new RamTensor({ 3,8,1 }, flt);
  Tensor in2 = new RomTensor({ 8,1 }, flt, s_ref_div_in2_05);
  Tensor ref_out = new RomTensor({ 3,8,1 }, flt, s_ref_div_out_05);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 24; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 25
 ***************************************/
TEST(ReferenceSub, random_gen_sub__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4050*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 25,9,18 }, flt, s_ref_sub_out_06);
  Tensor in2 = new RomTensor({ 18 }, flt, s_ref_sub_in2_06);
  Tensor out = new RamTensor({ 25,9,18 }, flt);
  Tensor in1 = new RomTensor({ 25,9,18 }, flt, s_ref_sub_in1_06);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4050; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 26
 ***************************************/
TEST(ReferenceAdd, random_gen_add__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<385*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 7,11,1,5 }, flt);
  Tensor in1 = new RomTensor({ 7,11,1,5 }, flt, s_ref_add_in1_06);
  Tensor ref_out = new RomTensor({ 7,11,1,5 }, flt, s_ref_add_out_06);
  Tensor in2 = new RomTensor({ 11,1,5 }, flt, s_ref_add_in2_06);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 385; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 27
 ***************************************/
TEST(ReferenceMul, random_gen_mul__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<7700*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 25,14,22 }, flt);
  Tensor in2 = new RomTensor({ 22 }, flt, s_ref_mul_in2_06);
  Tensor in1 = new RomTensor({ 25,14,22 }, flt, s_ref_mul_in1_06);
  Tensor ref_out = new RomTensor({ 25,14,22 }, flt, s_ref_mul_out_06);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 7700; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 28
 ***************************************/
TEST(ReferenceDiv, random_gen_div__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1250*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 25 }, flt, s_ref_div_in2_06);
  Tensor out = new RamTensor({ 25,2,25 }, flt);
  Tensor in1 = new RomTensor({ 25,2,25 }, flt, s_ref_div_in1_06);
  Tensor ref_out = new RomTensor({ 25,2,25 }, flt, s_ref_div_out_06);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1250; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 29
 ***************************************/
TEST(ReferenceSub, random_gen_sub__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<105*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 5,1 }, flt, s_ref_sub_in2_07);
  Tensor in1 = new RomTensor({ 21,5,1 }, flt, s_ref_sub_in1_07);
  Tensor out = new RamTensor({ 21,5,1 }, flt);
  Tensor ref_out = new RomTensor({ 21,5,1 }, flt, s_ref_sub_out_07);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 105; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 30
 ***************************************/
TEST(ReferenceAdd, random_gen_add__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<384*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 2,8,1,24 }, flt);
  Tensor in1 = new RomTensor({ 2,8,1,24 }, flt, s_ref_add_in1_07);
  Tensor in2 = new RomTensor({ 1,24 }, flt, s_ref_add_in2_07);
  Tensor ref_out = new RomTensor({ 2,8,1,24 }, flt, s_ref_add_out_07);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 384; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 31
 ***************************************/
TEST(ReferenceMul, random_gen_mul__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1260*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 21,10,1,6 }, flt, s_ref_mul_out_07);
  Tensor in2 = new RomTensor({ 6 }, flt, s_ref_mul_in2_07);
  Tensor in1 = new RomTensor({ 21,10,1,6 }, flt, s_ref_mul_in1_07);
  Tensor out = new RamTensor({ 21,10,1,6 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1260; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 32
 ***************************************/
TEST(ReferenceDiv, random_gen_div__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<3000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 20,15,1,10 }, flt);
  Tensor in2 = new RomTensor({ 1,10 }, flt, s_ref_div_in2_07);
  Tensor ref_out = new RomTensor({ 20,15,1,10 }, flt, s_ref_div_out_07);
  Tensor in1 = new RomTensor({ 20,15,1,10 }, flt, s_ref_div_in1_07);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 3000; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 33
 ***************************************/
TEST(ReferenceSub, random_gen_sub__08) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<10368*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 18,24,24 }, flt, s_ref_sub_out_08);
  Tensor in2 = new RomTensor({ 24 }, flt, s_ref_sub_in2_08);
  Tensor out = new RamTensor({ 18,24,24 }, flt);
  Tensor in1 = new RomTensor({ 18,24,24 }, flt, s_ref_sub_in1_08);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 10368; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 34
 ***************************************/
TEST(ReferenceAdd, random_gen_add__08) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1380*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 23,12,5 }, flt, s_ref_add_in1_08);
  Tensor ref_out = new RomTensor({ 23,12,5 }, flt, s_ref_add_out_08);
  Tensor out = new RamTensor({ 23,12,5 }, flt);
  Tensor in2 = new RomTensor({ 5 }, flt, s_ref_add_in2_08);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1380; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 35
 ***************************************/
TEST(ReferenceMul, random_gen_mul__08) {
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

  Tensor in1 = new RomTensor({ 5,1,2 }, flt, s_ref_mul_in1_08);
  Tensor ref_out = new RomTensor({ 5,1,2 }, flt, s_ref_mul_out_08);
  Tensor in2 = new RomTensor({ 1,2 }, flt, s_ref_mul_in2_08);
  Tensor out = new RamTensor({ 5,1,2 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 10; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 36
 ***************************************/
TEST(ReferenceDiv, random_gen_div__08) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<3780*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 9,21,1,20 }, flt);
  Tensor in1 = new RomTensor({ 9,21,1,20 }, flt, s_ref_div_in1_08);
  Tensor in2 = new RomTensor({ 1,20 }, flt, s_ref_div_in2_08);
  Tensor ref_out = new RomTensor({ 9,21,1,20 }, flt, s_ref_div_out_08);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 3780; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 37
 ***************************************/
TEST(ReferenceSub, random_gen_sub__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5670*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 18,15,21 }, flt, s_ref_sub_out_09);
  Tensor in1 = new RomTensor({ 18,15,21 }, flt, s_ref_sub_in1_09);
  Tensor in2 = new RomTensor({ 21 }, flt, s_ref_sub_in2_09);
  Tensor out = new RamTensor({ 18,15,21 }, flt);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5670; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 38
 ***************************************/
TEST(ReferenceAdd, random_gen_add__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<264*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 22,1,12 }, flt, s_ref_add_in1_09);
  Tensor in2 = new RomTensor({ 1,12 }, flt, s_ref_add_in2_09);
  Tensor out = new RamTensor({ 22,1,12 }, flt);
  Tensor ref_out = new RomTensor({ 22,1,12 }, flt, s_ref_add_out_09);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 264; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 39
 ***************************************/
TEST(ReferenceMul, random_gen_mul__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<330*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 15,22,1 }, flt, s_ref_mul_in1_09);
  Tensor ref_out = new RomTensor({ 15,22,1 }, flt, s_ref_mul_out_09);
  Tensor out = new RamTensor({ 15,22,1 }, flt);
  Tensor in2 = new RomTensor({ 22,1 }, flt, s_ref_mul_in2_09);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 330; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 40
 ***************************************/
TEST(ReferenceDiv, random_gen_div__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<68*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 17,4,1 }, flt, s_ref_div_in2_09);
  Tensor out = new RamTensor({ 1,17,4,1 }, flt);
  Tensor in1 = new RomTensor({ 1,17,4,1 }, flt, s_ref_div_in1_09);
  Tensor ref_out = new RomTensor({ 1,17,4,1 }, flt, s_ref_div_out_09);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 68; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 41
 ***************************************/
TEST(ReferenceSub, random_gen_sub__10) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2016*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 14,6,1,24 }, flt, s_ref_sub_out_10);
  Tensor in1 = new RomTensor({ 14,6,1,24 }, flt, s_ref_sub_in1_10);
  Tensor in2 = new RomTensor({ 6,1,24 }, flt, s_ref_sub_in2_10);
  Tensor out = new RamTensor({ 14,6,1,24 }, flt);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 2016; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 42
 ***************************************/
TEST(ReferenceAdd, random_gen_add__10) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<350*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 2,7,25 }, flt, s_ref_add_in1_10);
  Tensor ref_out = new RomTensor({ 2,7,25 }, flt, s_ref_add_out_10);
  Tensor in2 = new RomTensor({ 25 }, flt, s_ref_add_in2_10);
  Tensor out = new RamTensor({ 2,7,25 }, flt);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 350; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 43
 ***************************************/
TEST(ReferenceMul, random_gen_mul__10) {
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

  Tensor ref_out = new RomTensor({ 10,14,1 }, flt, s_ref_mul_out_10);
  Tensor in2 = new RomTensor({ 14,1 }, flt, s_ref_mul_in2_10);
  Tensor out = new RamTensor({ 10,14,1 }, flt);
  Tensor in1 = new RomTensor({ 10,14,1 }, flt, s_ref_mul_in1_10);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 140; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 44
 ***************************************/
TEST(ReferenceDiv, random_gen_div__10) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<76*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 4,1,19 }, flt);
  Tensor ref_out = new RomTensor({ 4,1,19 }, flt, s_ref_div_out_10);
  Tensor in1 = new RomTensor({ 4,1,19 }, flt, s_ref_div_in1_10);
  Tensor in2 = new RomTensor({ 1,19 }, flt, s_ref_div_in2_10);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 76; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 45
 ***************************************/
TEST(ReferenceSub, random_gen_sub__11) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<500*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 20,25,1,1 }, flt, s_ref_sub_in1_11);
  Tensor out = new RamTensor({ 20,25,1,1 }, flt);
  Tensor in2 = new RomTensor({ 25,1,1 }, flt, s_ref_sub_in2_11);
  Tensor ref_out = new RomTensor({ 20,25,1,1 }, flt, s_ref_sub_out_11);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 500; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 46
 ***************************************/
TEST(ReferenceAdd, random_gen_add__11) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<170*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 17,1,10 }, flt, s_ref_add_out_11);
  Tensor in1 = new RomTensor({ 17,1,10 }, flt, s_ref_add_in1_11);
  Tensor out = new RamTensor({ 17,1,10 }, flt);
  Tensor in2 = new RomTensor({ 1,10 }, flt, s_ref_add_in2_11);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 170; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 47
 ***************************************/
TEST(ReferenceMul, random_gen_mul__11) {
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

  Tensor ref_out = new RomTensor({ 8,21,24,1 }, flt, s_ref_mul_out_11);
  Tensor out = new RamTensor({ 8,21,24,1 }, flt);
  Tensor in2 = new RomTensor({ 24,1 }, flt, s_ref_mul_in2_11);
  Tensor in1 = new RomTensor({ 8,21,24,1 }, flt, s_ref_mul_in1_11);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4032; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 48
 ***************************************/
TEST(ReferenceDiv, random_gen_div__11) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<7056*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 14,21,24 }, flt, s_ref_div_in1_11);
  Tensor ref_out = new RomTensor({ 14,21,24 }, flt, s_ref_div_out_11);
  Tensor out = new RamTensor({ 14,21,24 }, flt);
  Tensor in2 = new RomTensor({ 24 }, flt, s_ref_div_in2_11);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 7056; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 49
 ***************************************/
TEST(ReferenceSub, random_gen_sub__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<36*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 12,3,1 }, flt);
  Tensor in2 = new RomTensor({ 3,1 }, flt, s_ref_sub_in2_12);
  Tensor in1 = new RomTensor({ 12,3,1 }, flt, s_ref_sub_in1_12);
  Tensor ref_out = new RomTensor({ 12,3,1 }, flt, s_ref_sub_out_12);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 36; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 50
 ***************************************/
TEST(ReferenceAdd, random_gen_add__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5290*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 10,23,1 }, flt, s_ref_add_in2_12);
  Tensor out = new RamTensor({ 23,10,23,1 }, flt);
  Tensor ref_out = new RomTensor({ 23,10,23,1 }, flt, s_ref_add_out_12);
  Tensor in1 = new RomTensor({ 23,10,23,1 }, flt, s_ref_add_in1_12);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5290; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 51
 ***************************************/
TEST(ReferenceMul, random_gen_mul__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<182*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_mul_in2_12);
  Tensor out = new RamTensor({ 14,13,1 }, flt);
  Tensor in1 = new RomTensor({ 14,13,1 }, flt, s_ref_mul_in1_12);
  Tensor ref_out = new RomTensor({ 14,13,1 }, flt, s_ref_mul_out_12);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 182; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 52
 ***************************************/
TEST(ReferenceDiv, random_gen_div__12) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<44000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 8,20,25,11 }, flt, s_ref_div_in1_12);
  Tensor in2 = new RomTensor({ 11 }, flt, s_ref_div_in2_12);
  Tensor ref_out = new RomTensor({ 8,20,25,11 }, flt, s_ref_div_out_12);
  Tensor out = new RamTensor({ 8,20,25,11 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 44000; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 53
 ***************************************/
TEST(ReferenceSub, random_gen_sub__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<253*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 23,1 }, flt, s_ref_sub_in2_13);
  Tensor out = new RamTensor({ 11,23,1 }, flt);
  Tensor in1 = new RomTensor({ 11,23,1 }, flt, s_ref_sub_in1_13);
  Tensor ref_out = new RomTensor({ 11,23,1 }, flt, s_ref_sub_out_13);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 253; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 54
 ***************************************/
TEST(ReferenceAdd, random_gen_add__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<180*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 12,15,1 }, flt, s_ref_add_in1_13);
  Tensor out = new RamTensor({ 12,15,1 }, flt);
  Tensor ref_out = new RomTensor({ 12,15,1 }, flt, s_ref_add_out_13);
  Tensor in2 = new RomTensor({ 15,1 }, flt, s_ref_add_in2_13);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 180; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 55
 ***************************************/
TEST(ReferenceMul, random_gen_mul__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<168*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,24 }, flt, s_ref_mul_in2_13);
  Tensor ref_out = new RomTensor({ 7,1,24 }, flt, s_ref_mul_out_13);
  Tensor in1 = new RomTensor({ 7,1,24 }, flt, s_ref_mul_in1_13);
  Tensor out = new RamTensor({ 7,1,24 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 168; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 56
 ***************************************/
TEST(ReferenceDiv, random_gen_div__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<240*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 16,1 }, flt, s_ref_div_in2_13);
  Tensor out = new RamTensor({ 15,16,1 }, flt);
  Tensor in1 = new RomTensor({ 15,16,1 }, flt, s_ref_div_in1_13);
  Tensor ref_out = new RomTensor({ 15,16,1 }, flt, s_ref_div_out_13);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 240; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 57
 ***************************************/
TEST(ReferenceSub, random_gen_sub__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<46000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 10,25,8,23 }, flt);
  Tensor in2 = new RomTensor({ 23 }, flt, s_ref_sub_in2_14);
  Tensor ref_out = new RomTensor({ 10,25,8,23 }, flt, s_ref_sub_out_14);
  Tensor in1 = new RomTensor({ 10,25,8,23 }, flt, s_ref_sub_in1_14);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 46000; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 58
 ***************************************/
TEST(ReferenceAdd, random_gen_add__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5850*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 13 }, flt, s_ref_add_in2_14);
  Tensor out = new RamTensor({ 25,18,13 }, flt);
  Tensor in1 = new RomTensor({ 25,18,13 }, flt, s_ref_add_in1_14);
  Tensor ref_out = new RomTensor({ 25,18,13 }, flt, s_ref_add_out_14);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5850; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 59
 ***************************************/
TEST(ReferenceMul, random_gen_mul__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<28*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 2,14,1 }, flt, s_ref_mul_out_14);
  Tensor in1 = new RomTensor({ 2,14,1 }, flt, s_ref_mul_in1_14);
  Tensor in2 = new RomTensor({ 14,1 }, flt, s_ref_mul_in2_14);
  Tensor out = new RamTensor({ 2,14,1 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 28; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 60
 ***************************************/
TEST(ReferenceDiv, random_gen_div__14) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2800*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 14,8,1,25 }, flt, s_ref_div_in1_14);
  Tensor ref_out = new RomTensor({ 14,8,1,25 }, flt, s_ref_div_out_14);
  Tensor in2 = new RomTensor({ 1,25 }, flt, s_ref_div_in2_14);
  Tensor out = new RamTensor({ 14,8,1,25 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 2800; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 61
 ***************************************/
TEST(ReferenceSub, random_gen_sub__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5082*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 11,7,6,11 }, flt);
  Tensor in2 = new RomTensor({ 11 }, flt, s_ref_sub_in2_15);
  Tensor ref_out = new RomTensor({ 11,7,6,11 }, flt, s_ref_sub_out_15);
  Tensor in1 = new RomTensor({ 11,7,6,11 }, flt, s_ref_sub_in1_15);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5082; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 62
 ***************************************/
TEST(ReferenceAdd, random_gen_add__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1728*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 8,18,12 }, flt, s_ref_add_in1_15);
  Tensor out = new RamTensor({ 8,18,12 }, flt);
  Tensor in2 = new RomTensor({ 12 }, flt, s_ref_add_in2_15);
  Tensor ref_out = new RomTensor({ 8,18,12 }, flt, s_ref_add_out_15);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1728; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 63
 ***************************************/
TEST(ReferenceMul, random_gen_mul__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 10,10,12 }, flt, s_ref_mul_out_15);
  Tensor in1 = new RomTensor({ 10,10,12 }, flt, s_ref_mul_in1_15);
  Tensor in2 = new RomTensor({ 12 }, flt, s_ref_mul_in2_15);
  Tensor out = new RamTensor({ 10,10,12 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1200; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 64
 ***************************************/
TEST(ReferenceDiv, random_gen_div__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<352*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 11,2,1,16 }, flt, s_ref_div_in1_15);
  Tensor ref_out = new RomTensor({ 11,2,1,16 }, flt, s_ref_div_out_15);
  Tensor in2 = new RomTensor({ 1,16 }, flt, s_ref_div_in2_15);
  Tensor out = new RamTensor({ 11,2,1,16 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 352; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 65
 ***************************************/
TEST(ReferenceSub, random_gen_sub__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<60*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 15,2,2,1 }, flt, s_ref_sub_in1_16);
  Tensor out = new RamTensor({ 15,2,2,1 }, flt);
  Tensor in2 = new RomTensor({ 2,1 }, flt, s_ref_sub_in2_16);
  Tensor ref_out = new RomTensor({ 15,2,2,1 }, flt, s_ref_sub_out_16);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 60; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 66
 ***************************************/
TEST(ReferenceAdd, random_gen_add__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<15960*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 19,14,15,4 }, flt, s_ref_add_in1_16);
  Tensor ref_out = new RomTensor({ 19,14,15,4 }, flt, s_ref_add_out_16);
  Tensor in2 = new RomTensor({ 4 }, flt, s_ref_add_in2_16);
  Tensor out = new RamTensor({ 19,14,15,4 }, flt);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 15960; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 67
 ***************************************/
TEST(ReferenceMul, random_gen_mul__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<3000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 25,8,15,1 }, flt);
  Tensor ref_out = new RomTensor({ 25,8,15,1 }, flt, s_ref_mul_out_16);
  Tensor in1 = new RomTensor({ 25,8,15,1 }, flt, s_ref_mul_in1_16);
  Tensor in2 = new RomTensor({ 15,1 }, flt, s_ref_mul_in2_16);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 3000; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 68
 ***************************************/
TEST(ReferenceDiv, random_gen_div__16) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<121*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_div_in2_16);
  Tensor in1 = new RomTensor({ 11,11,1 }, flt, s_ref_div_in1_16);
  Tensor ref_out = new RomTensor({ 11,11,1 }, flt, s_ref_div_out_16);
  Tensor out = new RamTensor({ 11,11,1 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 121; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 69
 ***************************************/
TEST(ReferenceSub, random_gen_sub__17) {
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

  Tensor in1 = new RomTensor({ 4,1,5 }, flt, s_ref_sub_in1_17);
  Tensor in2 = new RomTensor({ 1,5 }, flt, s_ref_sub_in2_17);
  Tensor ref_out = new RomTensor({ 4,1,5 }, flt, s_ref_sub_out_17);
  Tensor out = new RamTensor({ 4,1,5 }, flt);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 20; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 70
 ***************************************/
TEST(ReferenceAdd, random_gen_add__17) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1400*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 20 }, flt, s_ref_add_in2_17);
  Tensor out = new RamTensor({ 10,7,20 }, flt);
  Tensor in1 = new RomTensor({ 10,7,20 }, flt, s_ref_add_in1_17);
  Tensor ref_out = new RomTensor({ 10,7,20 }, flt, s_ref_add_out_17);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1400; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 71
 ***************************************/
TEST(ReferenceMul, random_gen_mul__17) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<495*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 5,9,1,11 }, flt, s_ref_mul_out_17);
  Tensor in2 = new RomTensor({ 1,11 }, flt, s_ref_mul_in2_17);
  Tensor in1 = new RomTensor({ 5,9,1,11 }, flt, s_ref_mul_in1_17);
  Tensor out = new RamTensor({ 5,9,1,11 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 495; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 72
 ***************************************/
TEST(ReferenceDiv, random_gen_div__17) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<11592*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 24,1,23,21 }, flt, s_ref_div_in1_17);
  Tensor out = new RamTensor({ 24,1,23,21 }, flt);
  Tensor ref_out = new RomTensor({ 24,1,23,21 }, flt, s_ref_div_out_17);
  Tensor in2 = new RomTensor({ 1,23,21 }, flt, s_ref_div_in2_17);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 11592; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 73
 ***************************************/
TEST(ReferenceSub, random_gen_sub__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<6762*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 14,21,1,23 }, flt, s_ref_sub_in1_18);
  Tensor in2 = new RomTensor({ 21,1,23 }, flt, s_ref_sub_in2_18);
  Tensor out = new RamTensor({ 14,21,1,23 }, flt);
  Tensor ref_out = new RomTensor({ 14,21,1,23 }, flt, s_ref_sub_out_18);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 6762; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 74
 ***************************************/
TEST(ReferenceAdd, random_gen_add__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<72*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 8,1 }, flt, s_ref_add_in2_18);
  Tensor out = new RamTensor({ 9,8,1 }, flt);
  Tensor in1 = new RomTensor({ 9,8,1 }, flt, s_ref_add_in1_18);
  Tensor ref_out = new RomTensor({ 9,8,1 }, flt, s_ref_add_out_18);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 72; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 75
 ***************************************/
TEST(ReferenceMul, random_gen_mul__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<18360*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 15,4,18,17 }, flt, s_ref_mul_out_18);
  Tensor in1 = new RomTensor({ 15,4,18,17 }, flt, s_ref_mul_in1_18);
  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_mul_in2_18);
  Tensor out = new RamTensor({ 15,4,18,17 }, flt);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 18360; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 76
 ***************************************/
TEST(ReferenceDiv, random_gen_div__18) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1134*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 9,7,1,18 }, flt, s_ref_div_out_18);
  Tensor in1 = new RomTensor({ 9,7,1,18 }, flt, s_ref_div_in1_18);
  Tensor in2 = new RomTensor({ 1,18 }, flt, s_ref_div_in2_18);
  Tensor out = new RamTensor({ 9,7,1,18 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1134; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 77
 ***************************************/
TEST(ReferenceSub, random_gen_sub__19) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4140*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 12,15,23,1 }, flt, s_ref_sub_in1_19);
  Tensor ref_out = new RomTensor({ 12,15,23,1 }, flt, s_ref_sub_out_19);
  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_sub_in2_19);
  Tensor out = new RamTensor({ 12,15,23,1 }, flt);

  ReferenceOperators::SubOperator<float> sub_op;
  sub_op
  .set_inputs({ 
    { ReferenceOperators::SubOperator<float>::a, in1 },
    { ReferenceOperators::SubOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::SubOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4140; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 78
 ***************************************/
TEST(ReferenceAdd, random_gen_add__19) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<17640*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 7,15,24,7 }, flt, s_ref_add_in1_19);
  Tensor ref_out = new RomTensor({ 7,15,24,7 }, flt, s_ref_add_out_19);
  Tensor out = new RamTensor({ 7,15,24,7 }, flt);
  Tensor in2 = new RomTensor({ 7 }, flt, s_ref_add_in2_19);

  ReferenceOperators::AddOperator<float> add_op;
  add_op
  .set_inputs({ 
    { ReferenceOperators::AddOperator<float>::a, in1 },
    { ReferenceOperators::AddOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::AddOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 17640; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 79
 ***************************************/
TEST(ReferenceMul, random_gen_mul__19) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1260*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 21 }, flt, s_ref_mul_in2_19);
  Tensor out = new RamTensor({ 4,15,21 }, flt);
  Tensor ref_out = new RomTensor({ 4,15,21 }, flt, s_ref_mul_out_19);
  Tensor in1 = new RomTensor({ 4,15,21 }, flt, s_ref_mul_in1_19);

  ReferenceOperators::MulOperator<float> mul_op;
  mul_op
  .set_inputs({ 
    { ReferenceOperators::MulOperator<float>::a, in1 },
    { ReferenceOperators::MulOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::MulOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1260; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

/***************************************
 * Generated Test 80
 ***************************************/
TEST(ReferenceDiv, random_gen_div__19) {
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

  Tensor in1 = new RomTensor({ 11,1,4 }, flt, s_ref_div_in1_19);
  Tensor in2 = new RomTensor({ 1,4 }, flt, s_ref_div_in2_19);
  Tensor ref_out = new RomTensor({ 11,1,4 }, flt, s_ref_div_out_19);
  Tensor out = new RamTensor({ 11,1,4 }, flt);

  ReferenceOperators::DivOperator<float> div_op;
  div_op
  .set_inputs({ 
    { ReferenceOperators::DivOperator<float>::a, in1 },
    { ReferenceOperators::DivOperator<float>::b, in2 }
  }).set_outputs({ 
    { ReferenceOperators::DivOperator<float>::c, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 44; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

