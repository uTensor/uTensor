#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_arithmetic.hpp"
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
  localCircularArenaAllocator<6240*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 12,13,2,20 }, flt, s_ref_sub_in1_00);
  Tensor ref_out = new RomTensor({ 12,13,2,20 }, flt, s_ref_sub_out_00);
  Tensor out = new RamTensor({ 12,13,2,20 }, flt);
  Tensor in2 = new RomTensor({ 12,13,2,20 }, flt, s_ref_sub_in2_00);

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

  for(int i = 0; i < 6240; i++) {
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
  localCircularArenaAllocator<260*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 4,13,5 }, flt, s_ref_add_out_00);
  Tensor out = new RamTensor({ 4,13,5 }, flt);
  Tensor in2 = new RomTensor({ 4,13,5 }, flt, s_ref_add_in2_00);
  Tensor in1 = new RomTensor({ 4,13,5 }, flt, s_ref_add_in1_00);

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

  for(int i = 0; i < 260; i++) {
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
  localCircularArenaAllocator<440*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 22,20 }, flt, s_ref_mul_out_00);
  Tensor in2 = new RomTensor({ 22,20 }, flt, s_ref_mul_in2_00);
  Tensor in1 = new RomTensor({ 22,20 }, flt, s_ref_mul_in1_00);
  Tensor out = new RamTensor({ 22,20 }, flt);

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

  for(int i = 0; i < 440; i++) {
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
  localCircularArenaAllocator<4368*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 24,14,13 }, flt);
  Tensor ref_out = new RomTensor({ 24,14,13 }, flt, s_ref_div_out_00);
  Tensor in1 = new RomTensor({ 24,14,13 }, flt, s_ref_div_in1_00);
  Tensor in2 = new RomTensor({ 24,14,13 }, flt, s_ref_div_in2_00);

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

  for(int i = 0; i < 4368; i++) {
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
  localCircularArenaAllocator<2070*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 23,18,5 }, flt);
  Tensor in2 = new RomTensor({ 23,18,5 }, flt, s_ref_sub_in2_01);
  Tensor ref_out = new RomTensor({ 23,18,5 }, flt, s_ref_sub_out_01);
  Tensor in1 = new RomTensor({ 23,18,5 }, flt, s_ref_sub_in1_01);

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

  for(int i = 0; i < 2070; i++) {
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
  localCircularArenaAllocator<171*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 9,19 }, flt);
  Tensor in1 = new RomTensor({ 9,19 }, flt, s_ref_add_in1_01);
  Tensor in2 = new RomTensor({ 9,19 }, flt, s_ref_add_in2_01);
  Tensor ref_out = new RomTensor({ 9,19 }, flt, s_ref_add_out_01);

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

  for(int i = 0; i < 171; i++) {
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
  localCircularArenaAllocator<3402*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 9,21,18 }, flt, s_ref_mul_in1_01);
  Tensor ref_out = new RomTensor({ 9,21,18 }, flt, s_ref_mul_out_01);
  Tensor out = new RamTensor({ 9,21,18 }, flt);
  Tensor in2 = new RomTensor({ 9,21,18 }, flt, s_ref_mul_in2_01);

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

  for(int i = 0; i < 3402; i++) {
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
  localCircularArenaAllocator<1*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1 }, flt);
  Tensor ref_out = new RomTensor({ 1 }, flt, s_ref_div_out_01);
  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_div_in2_01);
  Tensor in1 = new RomTensor({ 1 }, flt, s_ref_div_in1_01);

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

  for(int i = 0; i < 1; i++) {
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
  localCircularArenaAllocator<250*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 25,10 }, flt, s_ref_sub_out_02);
  Tensor out = new RamTensor({ 25,10 }, flt);
  Tensor in1 = new RomTensor({ 25,10 }, flt, s_ref_sub_in1_02);
  Tensor in2 = new RomTensor({ 25,10 }, flt, s_ref_sub_in2_02);

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

  for(int i = 0; i < 250; i++) {
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
  localCircularArenaAllocator<22*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 22 }, flt, s_ref_add_out_02);
  Tensor out = new RamTensor({ 22 }, flt);
  Tensor in1 = new RomTensor({ 22 }, flt, s_ref_add_in1_02);
  Tensor in2 = new RomTensor({ 22 }, flt, s_ref_add_in2_02);

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

  for(int i = 0; i < 22; i++) {
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
  localCircularArenaAllocator<15*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_mul_in2_02);
  Tensor out = new RamTensor({ 15 }, flt);
  Tensor in1 = new RomTensor({ 15 }, flt, s_ref_mul_in1_02);
  Tensor ref_out = new RomTensor({ 15 }, flt, s_ref_mul_out_02);

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

  for(int i = 0; i < 15; i++) {
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
  localCircularArenaAllocator<275*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 5,11,5 }, flt, s_ref_div_in1_02);
  Tensor in2 = new RomTensor({ 5,11,5 }, flt, s_ref_div_in2_02);
  Tensor out = new RamTensor({ 5,11,5 }, flt);
  Tensor ref_out = new RomTensor({ 5,11,5 }, flt, s_ref_div_out_02);

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

  for(int i = 0; i < 275; i++) {
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
  localCircularArenaAllocator<4095*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 21,5,13,3 }, flt, s_ref_sub_in1_03);
  Tensor out = new RamTensor({ 21,5,13,3 }, flt);
  Tensor in2 = new RomTensor({ 21,5,13,3 }, flt, s_ref_sub_in2_03);
  Tensor ref_out = new RomTensor({ 21,5,13,3 }, flt, s_ref_sub_out_03);

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

  for(int i = 0; i < 4095; i++) {
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
  localCircularArenaAllocator<24276*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 21,17,4,17 }, flt, s_ref_add_in1_03);
  Tensor out = new RamTensor({ 21,17,4,17 }, flt);
  Tensor ref_out = new RomTensor({ 21,17,4,17 }, flt, s_ref_add_out_03);
  Tensor in2 = new RomTensor({ 21,17,4,17 }, flt, s_ref_add_in2_03);

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

  for(int i = 0; i < 24276; i++) {
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
  localCircularArenaAllocator<42*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 6,7 }, flt, s_ref_mul_out_03);
  Tensor in2 = new RomTensor({ 6,7 }, flt, s_ref_mul_in2_03);
  Tensor out = new RamTensor({ 6,7 }, flt);
  Tensor in1 = new RomTensor({ 6,7 }, flt, s_ref_mul_in1_03);

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

  for(int i = 0; i < 42; i++) {
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
  localCircularArenaAllocator<5760*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 12,20,24 }, flt, s_ref_div_in2_03);
  Tensor out = new RamTensor({ 12,20,24 }, flt);
  Tensor in1 = new RomTensor({ 12,20,24 }, flt, s_ref_div_in1_03);
  Tensor ref_out = new RomTensor({ 12,20,24 }, flt, s_ref_div_out_03);

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

  for(int i = 0; i < 5760; i++) {
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
  localCircularArenaAllocator<1408*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 16,11,8 }, flt, s_ref_sub_out_04);
  Tensor in1 = new RomTensor({ 16,11,8 }, flt, s_ref_sub_in1_04);
  Tensor out = new RamTensor({ 16,11,8 }, flt);
  Tensor in2 = new RomTensor({ 16,11,8 }, flt, s_ref_sub_in2_04);

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

  for(int i = 0; i < 1408; i++) {
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
  localCircularArenaAllocator<7*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 7,1 }, flt);
  Tensor ref_out = new RomTensor({ 7,1 }, flt, s_ref_add_out_04);
  Tensor in1 = new RomTensor({ 7,1 }, flt, s_ref_add_in1_04);
  Tensor in2 = new RomTensor({ 7,1 }, flt, s_ref_add_in2_04);

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

  for(int i = 0; i < 7; i++) {
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
  localCircularArenaAllocator<2*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 2 }, flt, s_ref_mul_in2_04);
  Tensor ref_out = new RomTensor({ 2 }, flt, s_ref_mul_out_04);
  Tensor in1 = new RomTensor({ 2 }, flt, s_ref_mul_in1_04);
  Tensor out = new RamTensor({ 2 }, flt);

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

  for(int i = 0; i < 2; i++) {
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
  localCircularArenaAllocator<132*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 11,12 }, flt, s_ref_div_in1_04);
  Tensor in2 = new RomTensor({ 11,12 }, flt, s_ref_div_in2_04);
  Tensor ref_out = new RomTensor({ 11,12 }, flt, s_ref_div_out_04);
  Tensor out = new RamTensor({ 11,12 }, flt);

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

  for(int i = 0; i < 132; i++) {
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
  localCircularArenaAllocator<25116*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 13,4,21,23 }, flt);
  Tensor in1 = new RomTensor({ 13,4,21,23 }, flt, s_ref_sub_in1_05);
  Tensor in2 = new RomTensor({ 13,4,21,23 }, flt, s_ref_sub_in2_05);
  Tensor ref_out = new RomTensor({ 13,4,21,23 }, flt, s_ref_sub_out_05);

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

  for(int i = 0; i < 25116; i++) {
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
  localCircularArenaAllocator<228*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 19,12 }, flt, s_ref_add_in2_05);
  Tensor ref_out = new RomTensor({ 19,12 }, flt, s_ref_add_out_05);
  Tensor in1 = new RomTensor({ 19,12 }, flt, s_ref_add_in1_05);
  Tensor out = new RamTensor({ 19,12 }, flt);

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

  for(int i = 0; i < 228; i++) {
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
  localCircularArenaAllocator<364*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 7,4,13 }, flt, s_ref_mul_in1_05);
  Tensor ref_out = new RomTensor({ 7,4,13 }, flt, s_ref_mul_out_05);
  Tensor in2 = new RomTensor({ 7,4,13 }, flt, s_ref_mul_in2_05);
  Tensor out = new RamTensor({ 7,4,13 }, flt);

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

  for(int i = 0; i < 364; i++) {
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
  localCircularArenaAllocator<3456*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 16,12,18 }, flt, s_ref_div_in1_05);
  Tensor ref_out = new RomTensor({ 16,12,18 }, flt, s_ref_div_out_05);
  Tensor out = new RamTensor({ 16,12,18 }, flt);
  Tensor in2 = new RomTensor({ 16,12,18 }, flt, s_ref_div_in2_05);

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

  for(int i = 0; i < 3456; i++) {
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
  localCircularArenaAllocator<23*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 23 }, flt, s_ref_sub_out_06);
  Tensor out = new RamTensor({ 23 }, flt);
  Tensor in1 = new RomTensor({ 23 }, flt, s_ref_sub_in1_06);
  Tensor in2 = new RomTensor({ 23 }, flt, s_ref_sub_in2_06);

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

  for(int i = 0; i < 23; i++) {
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
  localCircularArenaAllocator<528*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 6,22,4 }, flt);
  Tensor ref_out = new RomTensor({ 6,22,4 }, flt, s_ref_add_out_06);
  Tensor in1 = new RomTensor({ 6,22,4 }, flt, s_ref_add_in1_06);
  Tensor in2 = new RomTensor({ 6,22,4 }, flt, s_ref_add_in2_06);

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

  for(int i = 0; i < 528; i++) {
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
  localCircularArenaAllocator<234*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 13,3,6 }, flt);
  Tensor in1 = new RomTensor({ 13,3,6 }, flt, s_ref_mul_in1_06);
  Tensor in2 = new RomTensor({ 13,3,6 }, flt, s_ref_mul_in2_06);
  Tensor ref_out = new RomTensor({ 13,3,6 }, flt, s_ref_mul_out_06);

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

  for(int i = 0; i < 234; i++) {
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
  localCircularArenaAllocator<45360*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 18,20,9,14 }, flt, s_ref_div_in2_06);
  Tensor out = new RamTensor({ 18,20,9,14 }, flt);
  Tensor ref_out = new RomTensor({ 18,20,9,14 }, flt, s_ref_div_out_06);
  Tensor in1 = new RomTensor({ 18,20,9,14 }, flt, s_ref_div_in1_06);

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

  for(int i = 0; i < 45360; i++) {
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
  localCircularArenaAllocator<714*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 17,3,14 }, flt, s_ref_sub_in2_07);
  Tensor ref_out = new RomTensor({ 17,3,14 }, flt, s_ref_sub_out_07);
  Tensor in1 = new RomTensor({ 17,3,14 }, flt, s_ref_sub_in1_07);
  Tensor out = new RamTensor({ 17,3,14 }, flt);

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

  for(int i = 0; i < 714; i++) {
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
  localCircularArenaAllocator<360*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 9,20,2 }, flt, s_ref_add_out_07);
  Tensor in2 = new RomTensor({ 9,20,2 }, flt, s_ref_add_in2_07);
  Tensor in1 = new RomTensor({ 9,20,2 }, flt, s_ref_add_in1_07);
  Tensor out = new RamTensor({ 9,20,2 }, flt);

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

  for(int i = 0; i < 360; i++) {
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
  localCircularArenaAllocator<3600*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 6,4,6,25 }, flt, s_ref_mul_out_07);
  Tensor in1 = new RomTensor({ 6,4,6,25 }, flt, s_ref_mul_in1_07);
  Tensor in2 = new RomTensor({ 6,4,6,25 }, flt, s_ref_mul_in2_07);
  Tensor out = new RamTensor({ 6,4,6,25 }, flt);

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

  for(int i = 0; i < 3600; i++) {
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
  localCircularArenaAllocator<3*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 3 }, flt, s_ref_div_in2_07);
  Tensor out = new RamTensor({ 3 }, flt);
  Tensor in1 = new RomTensor({ 3 }, flt, s_ref_div_in1_07);
  Tensor ref_out = new RomTensor({ 3 }, flt, s_ref_div_out_07);

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

  for(int i = 0; i < 3; i++) {
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
  localCircularArenaAllocator<24840*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 4,18,15,23 }, flt, s_ref_sub_out_08);
  Tensor out = new RamTensor({ 4,18,15,23 }, flt);
  Tensor in1 = new RomTensor({ 4,18,15,23 }, flt, s_ref_sub_in1_08);
  Tensor in2 = new RomTensor({ 4,18,15,23 }, flt, s_ref_sub_in2_08);

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

  for(int i = 0; i < 24840; i++) {
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
  localCircularArenaAllocator<1428*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 17,12,7 }, flt, s_ref_add_in1_08);
  Tensor out = new RamTensor({ 17,12,7 }, flt);
  Tensor ref_out = new RomTensor({ 17,12,7 }, flt, s_ref_add_out_08);
  Tensor in2 = new RomTensor({ 17,12,7 }, flt, s_ref_add_in2_08);

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

  for(int i = 0; i < 1428; i++) {
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
  localCircularArenaAllocator<1782*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 22,9,9 }, flt, s_ref_mul_out_08);
  Tensor in1 = new RomTensor({ 22,9,9 }, flt, s_ref_mul_in1_08);
  Tensor in2 = new RomTensor({ 22,9,9 }, flt, s_ref_mul_in2_08);
  Tensor out = new RamTensor({ 22,9,9 }, flt);

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

  for(int i = 0; i < 1782; i++) {
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
  localCircularArenaAllocator<214200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 17,24,21,25 }, flt, s_ref_div_in1_08);
  Tensor out = new RamTensor({ 17,24,21,25 }, flt);
  Tensor ref_out = new RomTensor({ 17,24,21,25 }, flt, s_ref_div_out_08);
  Tensor in2 = new RomTensor({ 17,24,21,25 }, flt, s_ref_div_in2_08);

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

  for(int i = 0; i < 214200; i++) {
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
  localCircularArenaAllocator<80*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 8,10 }, flt, s_ref_sub_in1_09);
  Tensor ref_out = new RomTensor({ 8,10 }, flt, s_ref_sub_out_09);
  Tensor out = new RamTensor({ 8,10 }, flt);
  Tensor in2 = new RomTensor({ 8,10 }, flt, s_ref_sub_in2_09);

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

  for(int i = 0; i < 80; i++) {
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
  localCircularArenaAllocator<798*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 19,7,6 }, flt, s_ref_add_in2_09);
  Tensor ref_out = new RomTensor({ 19,7,6 }, flt, s_ref_add_out_09);
  Tensor in1 = new RomTensor({ 19,7,6 }, flt, s_ref_add_in1_09);
  Tensor out = new RamTensor({ 19,7,6 }, flt);

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

  for(int i = 0; i < 798; i++) {
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
  localCircularArenaAllocator<8*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 4,2 }, flt);
  Tensor in1 = new RomTensor({ 4,2 }, flt, s_ref_mul_in1_09);
  Tensor in2 = new RomTensor({ 4,2 }, flt, s_ref_mul_in2_09);
  Tensor ref_out = new RomTensor({ 4,2 }, flt, s_ref_mul_out_09);

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

  for(int i = 0; i < 8; i++) {
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
  localCircularArenaAllocator<741*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 13,19,3 }, flt, s_ref_div_in2_09);
  Tensor out = new RamTensor({ 13,19,3 }, flt);
  Tensor ref_out = new RomTensor({ 13,19,3 }, flt, s_ref_div_out_09);
  Tensor in1 = new RomTensor({ 13,19,3 }, flt, s_ref_div_in1_09);

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

  for(int i = 0; i < 741; i++) {
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
  localCircularArenaAllocator<54720*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 19,20,9,16 }, flt, s_ref_sub_out_10);
  Tensor in1 = new RomTensor({ 19,20,9,16 }, flt, s_ref_sub_in1_10);
  Tensor in2 = new RomTensor({ 19,20,9,16 }, flt, s_ref_sub_in2_10);
  Tensor out = new RamTensor({ 19,20,9,16 }, flt);

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

  for(int i = 0; i < 54720; i++) {
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
  localCircularArenaAllocator<4048*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 16,11,23 }, flt, s_ref_add_in2_10);
  Tensor in1 = new RomTensor({ 16,11,23 }, flt, s_ref_add_in1_10);
  Tensor out = new RamTensor({ 16,11,23 }, flt);
  Tensor ref_out = new RomTensor({ 16,11,23 }, flt, s_ref_add_out_10);

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

  for(int i = 0; i < 4048; i++) {
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
  localCircularArenaAllocator<13200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 25,2,24,11 }, flt, s_ref_mul_in1_10);
  Tensor out = new RamTensor({ 25,2,24,11 }, flt);
  Tensor in2 = new RomTensor({ 25,2,24,11 }, flt, s_ref_mul_in2_10);
  Tensor ref_out = new RomTensor({ 25,2,24,11 }, flt, s_ref_mul_out_10);

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

  for(int i = 0; i < 13200; i++) {
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
  localCircularArenaAllocator<51*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,17,3 }, flt, s_ref_div_in2_10);
  Tensor ref_out = new RomTensor({ 1,17,3 }, flt, s_ref_div_out_10);
  Tensor in1 = new RomTensor({ 1,17,3 }, flt, s_ref_div_in1_10);
  Tensor out = new RamTensor({ 1,17,3 }, flt);

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

  for(int i = 0; i < 51; i++) {
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
  localCircularArenaAllocator<260*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 13,20 }, flt, s_ref_sub_in1_11);
  Tensor out = new RamTensor({ 13,20 }, flt);
  Tensor in2 = new RomTensor({ 13,20 }, flt, s_ref_sub_in2_11);
  Tensor ref_out = new RomTensor({ 13,20 }, flt, s_ref_sub_out_11);

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

  for(int i = 0; i < 260; i++) {
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
  localCircularArenaAllocator<16320*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 17,20,12,4 }, flt, s_ref_add_in1_11);
  Tensor in2 = new RomTensor({ 17,20,12,4 }, flt, s_ref_add_in2_11);
  Tensor out = new RamTensor({ 17,20,12,4 }, flt);
  Tensor ref_out = new RomTensor({ 17,20,12,4 }, flt, s_ref_add_out_11);

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

  for(int i = 0; i < 16320; i++) {
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
  localCircularArenaAllocator<6240*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 24,20,13 }, flt, s_ref_mul_in2_11);
  Tensor ref_out = new RomTensor({ 24,20,13 }, flt, s_ref_mul_out_11);
  Tensor in1 = new RomTensor({ 24,20,13 }, flt, s_ref_mul_in1_11);
  Tensor out = new RamTensor({ 24,20,13 }, flt);

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

  for(int i = 0; i < 6240; i++) {
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
  localCircularArenaAllocator<14*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 14 }, flt);
  Tensor ref_out = new RomTensor({ 14 }, flt, s_ref_div_out_11);
  Tensor in1 = new RomTensor({ 14 }, flt, s_ref_div_in1_11);
  Tensor in2 = new RomTensor({ 14 }, flt, s_ref_div_in2_11);

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

  for(int i = 0; i < 14; i++) {
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
  localCircularArenaAllocator<17*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 17 }, flt, s_ref_sub_out_12);
  Tensor out = new RamTensor({ 17 }, flt);
  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_sub_in2_12);
  Tensor in1 = new RomTensor({ 17 }, flt, s_ref_sub_in1_12);

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

  for(int i = 0; i < 17; i++) {
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
  localCircularArenaAllocator<51*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 3,17 }, flt);
  Tensor ref_out = new RomTensor({ 3,17 }, flt, s_ref_add_out_12);
  Tensor in2 = new RomTensor({ 3,17 }, flt, s_ref_add_in2_12);
  Tensor in1 = new RomTensor({ 3,17 }, flt, s_ref_add_in1_12);

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

  for(int i = 0; i < 51; i++) {
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
  localCircularArenaAllocator<1*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 1 }, flt, s_ref_mul_out_12);
  Tensor in1 = new RomTensor({ 1 }, flt, s_ref_mul_in1_12);
  Tensor out = new RamTensor({ 1 }, flt);
  Tensor in2 = new RomTensor({ 1 }, flt, s_ref_mul_in2_12);

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

  for(int i = 0; i < 1; i++) {
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
  localCircularArenaAllocator<2432*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 4,4,8,19 }, flt);
  Tensor in2 = new RomTensor({ 4,4,8,19 }, flt, s_ref_div_in2_12);
  Tensor in1 = new RomTensor({ 4,4,8,19 }, flt, s_ref_div_in1_12);
  Tensor ref_out = new RomTensor({ 4,4,8,19 }, flt, s_ref_div_out_12);

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

  for(int i = 0; i < 2432; i++) {
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
  localCircularArenaAllocator<8211*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 17,23,21 }, flt, s_ref_sub_in2_13);
  Tensor out = new RamTensor({ 17,23,21 }, flt);
  Tensor in1 = new RomTensor({ 17,23,21 }, flt, s_ref_sub_in1_13);
  Tensor ref_out = new RomTensor({ 17,23,21 }, flt, s_ref_sub_out_13);

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

  for(int i = 0; i < 8211; i++) {
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
  localCircularArenaAllocator<22*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 22 }, flt, s_ref_add_in2_13);
  Tensor ref_out = new RomTensor({ 22 }, flt, s_ref_add_out_13);
  Tensor out = new RamTensor({ 22 }, flt);
  Tensor in1 = new RomTensor({ 22 }, flt, s_ref_add_in1_13);

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

  for(int i = 0; i < 22; i++) {
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
  localCircularArenaAllocator<209*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 19,11 }, flt, s_ref_mul_out_13);
  Tensor in1 = new RomTensor({ 19,11 }, flt, s_ref_mul_in1_13);
  Tensor in2 = new RomTensor({ 19,11 }, flt, s_ref_mul_in2_13);
  Tensor out = new RamTensor({ 19,11 }, flt);

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

  for(int i = 0; i < 209; i++) {
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
  localCircularArenaAllocator<483*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 21,23 }, flt, s_ref_div_in1_13);
  Tensor out = new RamTensor({ 21,23 }, flt);
  Tensor in2 = new RomTensor({ 21,23 }, flt, s_ref_div_in2_13);
  Tensor ref_out = new RomTensor({ 21,23 }, flt, s_ref_div_out_13);

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

  for(int i = 0; i < 483; i++) {
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
  localCircularArenaAllocator<3600*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 18,2,10,10 }, flt, s_ref_sub_in2_14);
  Tensor out = new RamTensor({ 18,2,10,10 }, flt);
  Tensor ref_out = new RomTensor({ 18,2,10,10 }, flt, s_ref_sub_out_14);
  Tensor in1 = new RomTensor({ 18,2,10,10 }, flt, s_ref_sub_in1_14);

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

  for(int i = 0; i < 3600; i++) {
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
  localCircularArenaAllocator<1584*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 11,24,6 }, flt, s_ref_add_in2_14);
  Tensor ref_out = new RomTensor({ 11,24,6 }, flt, s_ref_add_out_14);
  Tensor out = new RamTensor({ 11,24,6 }, flt);
  Tensor in1 = new RomTensor({ 11,24,6 }, flt, s_ref_add_in1_14);

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

  for(int i = 0; i < 1584; i++) {
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
  localCircularArenaAllocator<75*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 25,3 }, flt, s_ref_mul_in1_14);
  Tensor ref_out = new RomTensor({ 25,3 }, flt, s_ref_mul_out_14);
  Tensor in2 = new RomTensor({ 25,3 }, flt, s_ref_mul_in2_14);
  Tensor out = new RamTensor({ 25,3 }, flt);

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

  for(int i = 0; i < 75; i++) {
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
  localCircularArenaAllocator<24*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 24 }, flt, s_ref_div_out_14);
  Tensor in2 = new RomTensor({ 24 }, flt, s_ref_div_in2_14);
  Tensor in1 = new RomTensor({ 24 }, flt, s_ref_div_in1_14);
  Tensor out = new RamTensor({ 24 }, flt);

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
 * Generated Test 61
 ***************************************/
TEST(ReferenceSub, random_gen_sub__15) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<33264*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 12,7,18,22 }, flt, s_ref_sub_in1_15);
  Tensor ref_out = new RomTensor({ 12,7,18,22 }, flt, s_ref_sub_out_15);
  Tensor in2 = new RomTensor({ 12,7,18,22 }, flt, s_ref_sub_in2_15);
  Tensor out = new RamTensor({ 12,7,18,22 }, flt);

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

  for(int i = 0; i < 33264; i++) {
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
  localCircularArenaAllocator<12960*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 16,6,9,15 }, flt, s_ref_add_in1_15);
  Tensor out = new RamTensor({ 16,6,9,15 }, flt);
  Tensor in2 = new RomTensor({ 16,6,9,15 }, flt, s_ref_add_in2_15);
  Tensor ref_out = new RomTensor({ 16,6,9,15 }, flt, s_ref_add_out_15);

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

  for(int i = 0; i < 12960; i++) {
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
  localCircularArenaAllocator<3213*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 21,17,9 }, flt, s_ref_mul_in2_15);
  Tensor in1 = new RomTensor({ 21,17,9 }, flt, s_ref_mul_in1_15);
  Tensor ref_out = new RomTensor({ 21,17,9 }, flt, s_ref_mul_out_15);
  Tensor out = new RamTensor({ 21,17,9 }, flt);

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

  for(int i = 0; i < 3213; i++) {
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
  localCircularArenaAllocator<144*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 8,3,6 }, flt, s_ref_div_in1_15);
  Tensor in2 = new RomTensor({ 8,3,6 }, flt, s_ref_div_in2_15);
  Tensor ref_out = new RomTensor({ 8,3,6 }, flt, s_ref_div_out_15);
  Tensor out = new RamTensor({ 8,3,6 }, flt);

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

  for(int i = 0; i < 144; i++) {
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
  localCircularArenaAllocator<30855*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 11,11,17,15 }, flt, s_ref_sub_in2_16);
  Tensor ref_out = new RomTensor({ 11,11,17,15 }, flt, s_ref_sub_out_16);
  Tensor out = new RamTensor({ 11,11,17,15 }, flt);
  Tensor in1 = new RomTensor({ 11,11,17,15 }, flt, s_ref_sub_in1_16);

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

  for(int i = 0; i < 30855; i++) {
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
  localCircularArenaAllocator<46189*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 19,11,17,13 }, flt);
  Tensor ref_out = new RomTensor({ 19,11,17,13 }, flt, s_ref_add_out_16);
  Tensor in2 = new RomTensor({ 19,11,17,13 }, flt, s_ref_add_in2_16);
  Tensor in1 = new RomTensor({ 19,11,17,13 }, flt, s_ref_add_in1_16);

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

  for(int i = 0; i < 46189; i++) {
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
  localCircularArenaAllocator<154*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 1,22,7 }, flt, s_ref_mul_out_16);
  Tensor in1 = new RomTensor({ 1,22,7 }, flt, s_ref_mul_in1_16);
  Tensor out = new RamTensor({ 1,22,7 }, flt);
  Tensor in2 = new RomTensor({ 1,22,7 }, flt, s_ref_mul_in2_16);

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

  for(int i = 0; i < 154; i++) {
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
  localCircularArenaAllocator<6*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 6 }, flt);
  Tensor in2 = new RomTensor({ 6 }, flt, s_ref_div_in2_16);
  Tensor in1 = new RomTensor({ 6 }, flt, s_ref_div_in1_16);
  Tensor ref_out = new RomTensor({ 6 }, flt, s_ref_div_out_16);

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

  for(int i = 0; i < 6; i++) {
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
  localCircularArenaAllocator<7*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 7 }, flt, s_ref_sub_in1_17);
  Tensor ref_out = new RomTensor({ 7 }, flt, s_ref_sub_out_17);
  Tensor in2 = new RomTensor({ 7 }, flt, s_ref_sub_in2_17);
  Tensor out = new RamTensor({ 7 }, flt);

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

  for(int i = 0; i < 7; i++) {
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
  localCircularArenaAllocator<12*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 12 }, flt, s_ref_add_in2_17);
  Tensor ref_out = new RomTensor({ 12 }, flt, s_ref_add_out_17);
  Tensor out = new RamTensor({ 12 }, flt);
  Tensor in1 = new RomTensor({ 12 }, flt, s_ref_add_in1_17);

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

  for(int i = 0; i < 12; i++) {
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
  localCircularArenaAllocator<345*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 23,15 }, flt);
  Tensor ref_out = new RomTensor({ 23,15 }, flt, s_ref_mul_out_17);
  Tensor in2 = new RomTensor({ 23,15 }, flt, s_ref_mul_in2_17);
  Tensor in1 = new RomTensor({ 23,15 }, flt, s_ref_mul_in1_17);

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

  for(int i = 0; i < 345; i++) {
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
  localCircularArenaAllocator<1012*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 11,23,4 }, flt, s_ref_div_in1_17);
  Tensor out = new RamTensor({ 11,23,4 }, flt);
  Tensor in2 = new RomTensor({ 11,23,4 }, flt, s_ref_div_in2_17);
  Tensor ref_out = new RomTensor({ 11,23,4 }, flt, s_ref_div_out_17);

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

  for(int i = 0; i < 1012; i++) {
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
  localCircularArenaAllocator<2754*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 18,9,17 }, flt, s_ref_sub_out_18);
  Tensor in2 = new RomTensor({ 18,9,17 }, flt, s_ref_sub_in2_18);
  Tensor in1 = new RomTensor({ 18,9,17 }, flt, s_ref_sub_in1_18);
  Tensor out = new RamTensor({ 18,9,17 }, flt);

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

  for(int i = 0; i < 2754; i++) {
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
  localCircularArenaAllocator<405*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 3,9,1,15 }, flt, s_ref_add_in2_18);
  Tensor out = new RamTensor({ 3,9,1,15 }, flt);
  Tensor ref_out = new RomTensor({ 3,9,1,15 }, flt, s_ref_add_out_18);
  Tensor in1 = new RomTensor({ 3,9,1,15 }, flt, s_ref_add_in1_18);

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

  for(int i = 0; i < 405; i++) {
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
  localCircularArenaAllocator<221*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 13,17 }, flt);
  Tensor in1 = new RomTensor({ 13,17 }, flt, s_ref_mul_in1_18);
  Tensor ref_out = new RomTensor({ 13,17 }, flt, s_ref_mul_out_18);
  Tensor in2 = new RomTensor({ 13,17 }, flt, s_ref_mul_in2_18);

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

  for(int i = 0; i < 221; i++) {
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
  localCircularArenaAllocator<81*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 9,1,9 }, flt);
  Tensor ref_out = new RomTensor({ 9,1,9 }, flt, s_ref_div_out_18);
  Tensor in1 = new RomTensor({ 9,1,9 }, flt, s_ref_div_in1_18);
  Tensor in2 = new RomTensor({ 9,1,9 }, flt, s_ref_div_in2_18);

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

  for(int i = 0; i < 81; i++) {
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
  localCircularArenaAllocator<392*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 7,4,2,7 }, flt, s_ref_sub_in2_19);
  Tensor out = new RamTensor({ 7,4,2,7 }, flt);
  Tensor ref_out = new RomTensor({ 7,4,2,7 }, flt, s_ref_sub_out_19);
  Tensor in1 = new RomTensor({ 7,4,2,7 }, flt, s_ref_sub_in1_19);

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

  for(int i = 0; i < 392; i++) {
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
  localCircularArenaAllocator<1320*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 6,10,22 }, flt, s_ref_add_in1_19);
  Tensor ref_out = new RomTensor({ 6,10,22 }, flt, s_ref_add_out_19);
  Tensor out = new RamTensor({ 6,10,22 }, flt);
  Tensor in2 = new RomTensor({ 6,10,22 }, flt, s_ref_add_in2_19);

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

  for(int i = 0; i < 1320; i++) {
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
  localCircularArenaAllocator<6*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 6 }, flt, s_ref_mul_in2_19);
  Tensor ref_out = new RomTensor({ 6 }, flt, s_ref_mul_out_19);
  Tensor in1 = new RomTensor({ 6 }, flt, s_ref_mul_in1_19);
  Tensor out = new RamTensor({ 6 }, flt);

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

  for(int i = 0; i < 6; i++) {
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
  localCircularArenaAllocator<945*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 9,21,5 }, flt, s_ref_div_in2_19);
  Tensor in1 = new RomTensor({ 9,21,5 }, flt, s_ref_div_in1_19);
  Tensor ref_out = new RomTensor({ 9,21,5 }, flt, s_ref_div_out_19);
  Tensor out = new RamTensor({ 9,21,5 }, flt);

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

  for(int i = 0; i < 945; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

