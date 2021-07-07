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
  localCircularArenaAllocator<96900*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 12,25,19,17 }, flt, s_ref_sub_out_00);
  Tensor in1 = new RomTensor({ 12,25,19,17 }, flt, s_ref_sub_in1_00);
  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_sub_in2_00);
  Tensor out = new RamTensor({ 12,25,19,17 }, flt);

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

  for(int i = 0; i < 96900; i++) {
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
  localCircularArenaAllocator<137088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 24,16,17,21 }, flt, s_ref_add_in1_00);
  Tensor in2 = new RomTensor({ 17,1 }, flt, s_ref_add_in2_00);
  Tensor ref_out = new RomTensor({ 24,16,17,21 }, flt, s_ref_add_out_00);
  Tensor out = new RamTensor({ 24,16,17,21 }, flt);

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

  for(int i = 0; i < 137088; i++) {
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
  localCircularArenaAllocator<396*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 3,12,11 }, flt);
  Tensor in2 = new RomTensor({ 1,11 }, flt, s_ref_mul_in2_00);
  Tensor ref_out = new RomTensor({ 3,12,11 }, flt, s_ref_mul_out_00);
  Tensor in1 = new RomTensor({ 3,12,11 }, flt, s_ref_mul_in1_00);

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

  for(int i = 0; i < 396; i++) {
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
  localCircularArenaAllocator<420*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 7,3,5,4 }, flt, s_ref_div_out_00);
  Tensor in1 = new RomTensor({ 7,3,5,4 }, flt, s_ref_div_in1_00);
  Tensor out = new RamTensor({ 7,3,5,4 }, flt);
  Tensor in2 = new RomTensor({ 5,4 }, flt, s_ref_div_in2_00);

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

  for(int i = 0; i < 420; i++) {
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
  localCircularArenaAllocator<476*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 4,7,17 }, flt);
  Tensor in1 = new RomTensor({ 4,7,17 }, flt, s_ref_sub_in1_01);
  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_sub_in2_01);
  Tensor ref_out = new RomTensor({ 4,7,17 }, flt, s_ref_sub_out_01);

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

  for(int i = 0; i < 476; i++) {
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
  localCircularArenaAllocator<252*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 21,2,6 }, flt);
  Tensor ref_out = new RomTensor({ 21,2,6 }, flt, s_ref_add_out_01);
  Tensor in2 = new RomTensor({ 6 }, flt, s_ref_add_in2_01);
  Tensor in1 = new RomTensor({ 21,2,6 }, flt, s_ref_add_in1_01);

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

  for(int i = 0; i < 252; i++) {
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
  localCircularArenaAllocator<6120*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 17,24,15 }, flt);
  Tensor in1 = new RomTensor({ 17,24,15 }, flt, s_ref_mul_in1_01);
  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_mul_in2_01);
  Tensor ref_out = new RomTensor({ 17,24,15 }, flt, s_ref_mul_out_01);

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

  for(int i = 0; i < 6120; i++) {
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
  localCircularArenaAllocator<240*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 15,8,2 }, flt);
  Tensor in2 = new RomTensor({ 2 }, flt, s_ref_div_in2_01);
  Tensor in1 = new RomTensor({ 15,8,2 }, flt, s_ref_div_in1_01);
  Tensor ref_out = new RomTensor({ 15,8,2 }, flt, s_ref_div_out_01);

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
 * Generated Test 9
 ***************************************/
TEST(ReferenceSub, random_gen_sub__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2310*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 11,15,14 }, flt);
  Tensor in2 = new RomTensor({ 14 }, flt, s_ref_sub_in2_02);
  Tensor ref_out = new RomTensor({ 11,15,14 }, flt, s_ref_sub_out_02);
  Tensor in1 = new RomTensor({ 11,15,14 }, flt, s_ref_sub_in1_02);

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

  for(int i = 0; i < 2310; i++) {
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
  localCircularArenaAllocator<5280*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 22,10,24 }, flt, s_ref_add_out_02);
  Tensor in2 = new RomTensor({ 10,24 }, flt, s_ref_add_in2_02);
  Tensor out = new RamTensor({ 22,10,24 }, flt);
  Tensor in1 = new RomTensor({ 22,10,24 }, flt, s_ref_add_in1_02);

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

  for(int i = 0; i < 5280; i++) {
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
  localCircularArenaAllocator<2040*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 17,4,6,5 }, flt, s_ref_mul_in1_02);
  Tensor in2 = new RomTensor({ 6,5 }, flt, s_ref_mul_in2_02);
  Tensor ref_out = new RomTensor({ 17,4,6,5 }, flt, s_ref_mul_out_02);
  Tensor out = new RamTensor({ 17,4,6,5 }, flt);

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

  for(int i = 0; i < 2040; i++) {
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
  localCircularArenaAllocator<425*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 17,5,5 }, flt, s_ref_div_out_02);
  Tensor in2 = new RomTensor({ 5 }, flt, s_ref_div_in2_02);
  Tensor out = new RamTensor({ 17,5,5 }, flt);
  Tensor in1 = new RomTensor({ 17,5,5 }, flt, s_ref_div_in1_02);

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

  for(int i = 0; i < 425; i++) {
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
  localCircularArenaAllocator<7938*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 9,3,21,14 }, flt);
  Tensor in1 = new RomTensor({ 9,3,21,14 }, flt, s_ref_sub_in1_03);
  Tensor ref_out = new RomTensor({ 9,3,21,14 }, flt, s_ref_sub_out_03);
  Tensor in2 = new RomTensor({ 14 }, flt, s_ref_sub_in2_03);

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

  for(int i = 0; i < 7938; i++) {
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
  localCircularArenaAllocator<1200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 5,12,20 }, flt, s_ref_add_out_03);
  Tensor out = new RamTensor({ 5,12,20 }, flt);
  Tensor in1 = new RomTensor({ 5,12,20 }, flt, s_ref_add_in1_03);
  Tensor in2 = new RomTensor({ 20 }, flt, s_ref_add_in2_03);

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

  for(int i = 0; i < 1200; i++) {
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
  localCircularArenaAllocator<3696*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 16,11,21 }, flt);
  Tensor in1 = new RomTensor({ 16,11,21 }, flt, s_ref_mul_in1_03);
  Tensor ref_out = new RomTensor({ 16,11,21 }, flt, s_ref_mul_out_03);
  Tensor in2 = new RomTensor({ 21 }, flt, s_ref_mul_in2_03);

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

  for(int i = 0; i < 3696; i++) {
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
  localCircularArenaAllocator<198720*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 24,24,23,15 }, flt);
  Tensor ref_out = new RomTensor({ 24,24,23,15 }, flt, s_ref_div_out_03);
  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_div_in2_03);
  Tensor in1 = new RomTensor({ 24,24,23,15 }, flt, s_ref_div_in1_03);

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

  for(int i = 0; i < 198720; i++) {
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
  localCircularArenaAllocator<84*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 2,2,21 }, flt, s_ref_sub_in1_04);
  Tensor out = new RamTensor({ 2,2,21 }, flt);
  Tensor in2 = new RomTensor({ 21 }, flt, s_ref_sub_in2_04);
  Tensor ref_out = new RomTensor({ 2,2,21 }, flt, s_ref_sub_out_04);

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

  for(int i = 0; i < 84; i++) {
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
  localCircularArenaAllocator<3570*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 17,21,10 }, flt, s_ref_add_out_04);
  Tensor in2 = new RomTensor({ 21,1 }, flt, s_ref_add_in2_04);
  Tensor out = new RamTensor({ 17,21,10 }, flt);
  Tensor in1 = new RomTensor({ 17,21,10 }, flt, s_ref_add_in1_04);

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

  for(int i = 0; i < 3570; i++) {
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

  Tensor in2 = new RomTensor({ 9 }, flt, s_ref_mul_in2_04);
  Tensor ref_out = new RomTensor({ 7,16,9 }, flt, s_ref_mul_out_04);
  Tensor in1 = new RomTensor({ 7,16,9 }, flt, s_ref_mul_in1_04);
  Tensor out = new RamTensor({ 7,16,9 }, flt);

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
  localCircularArenaAllocator<3200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 10,16,20 }, flt);
  Tensor in1 = new RomTensor({ 10,16,20 }, flt, s_ref_div_in1_04);
  Tensor in2 = new RomTensor({ 16,20 }, flt, s_ref_div_in2_04);
  Tensor ref_out = new RomTensor({ 10,16,20 }, flt, s_ref_div_out_04);

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

  for(int i = 0; i < 3200; i++) {
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
  localCircularArenaAllocator<10465*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 7 }, flt, s_ref_sub_in2_05);
  Tensor out = new RamTensor({ 23,5,13,7 }, flt);
  Tensor ref_out = new RomTensor({ 23,5,13,7 }, flt, s_ref_sub_out_05);
  Tensor in1 = new RomTensor({ 23,5,13,7 }, flt, s_ref_sub_in1_05);

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

  for(int i = 0; i < 10465; i++) {
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
  localCircularArenaAllocator<7020*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 15,3,13,12 }, flt, s_ref_add_out_05);
  Tensor in1 = new RomTensor({ 15,3,13,12 }, flt, s_ref_add_in1_05);
  Tensor out = new RamTensor({ 15,3,13,12 }, flt);
  Tensor in2 = new RomTensor({ 1,13,12 }, flt, s_ref_add_in2_05);

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

  for(int i = 0; i < 7020; i++) {
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
  localCircularArenaAllocator<78400*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 14 }, flt, s_ref_mul_in2_05);
  Tensor out = new RamTensor({ 25,16,14,14 }, flt);
  Tensor ref_out = new RomTensor({ 25,16,14,14 }, flt, s_ref_mul_out_05);
  Tensor in1 = new RomTensor({ 25,16,14,14 }, flt, s_ref_mul_in1_05);

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

  for(int i = 0; i < 78400; i++) {
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
  localCircularArenaAllocator<3570*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 21,17,10 }, flt);
  Tensor in1 = new RomTensor({ 21,17,10 }, flt, s_ref_div_in1_05);
  Tensor ref_out = new RomTensor({ 21,17,10 }, flt, s_ref_div_out_05);
  Tensor in2 = new RomTensor({ 1,10 }, flt, s_ref_div_in2_05);

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

  for(int i = 0; i < 3570; i++) {
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
  localCircularArenaAllocator<177744*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 23,14,23,24 }, flt);
  Tensor ref_out = new RomTensor({ 23,14,23,24 }, flt, s_ref_sub_out_06);
  Tensor in1 = new RomTensor({ 23,14,23,24 }, flt, s_ref_sub_in1_06);
  Tensor in2 = new RomTensor({ 24 }, flt, s_ref_sub_in2_06);

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

  for(int i = 0; i < 177744; i++) {
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
  localCircularArenaAllocator<188100*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 22,25,18,19 }, flt, s_ref_add_in1_06);
  Tensor ref_out = new RomTensor({ 22,25,18,19 }, flt, s_ref_add_out_06);
  Tensor in2 = new RomTensor({ 25,18,19 }, flt, s_ref_add_in2_06);
  Tensor out = new RamTensor({ 22,25,18,19 }, flt);

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

  for(int i = 0; i < 188100; i++) {
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
  localCircularArenaAllocator<920*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 8,23,5 }, flt);
  Tensor ref_out = new RomTensor({ 8,23,5 }, flt, s_ref_mul_out_06);
  Tensor in1 = new RomTensor({ 8,23,5 }, flt, s_ref_mul_in1_06);
  Tensor in2 = new RomTensor({ 23,5 }, flt, s_ref_mul_in2_06);

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

  for(int i = 0; i < 920; i++) {
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
  localCircularArenaAllocator<11856*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 1,19 }, flt, s_ref_div_in2_06);
  Tensor out = new RamTensor({ 13,4,12,19 }, flt);
  Tensor ref_out = new RomTensor({ 13,4,12,19 }, flt, s_ref_div_out_06);
  Tensor in1 = new RomTensor({ 13,4,12,19 }, flt, s_ref_div_in1_06);

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

  for(int i = 0; i < 11856; i++) {
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
  localCircularArenaAllocator<86020*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 10,22,17,23 }, flt, s_ref_sub_in1_07);
  Tensor ref_out = new RomTensor({ 10,22,17,23 }, flt, s_ref_sub_out_07);
  Tensor out = new RamTensor({ 10,22,17,23 }, flt);
  Tensor in2 = new RomTensor({ 23 }, flt, s_ref_sub_in2_07);

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

  for(int i = 0; i < 86020; i++) {
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
  localCircularArenaAllocator<78000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 25,12,20,13 }, flt, s_ref_add_out_07);
  Tensor in1 = new RomTensor({ 25,12,20,13 }, flt, s_ref_add_in1_07);
  Tensor in2 = new RomTensor({ 20,1 }, flt, s_ref_add_in2_07);
  Tensor out = new RamTensor({ 25,12,20,13 }, flt);

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

  for(int i = 0; i < 78000; i++) {
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
  localCircularArenaAllocator<4784*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 23,1 }, flt, s_ref_mul_in2_07);
  Tensor out = new RamTensor({ 13,23,16 }, flt);
  Tensor ref_out = new RomTensor({ 13,23,16 }, flt, s_ref_mul_out_07);
  Tensor in1 = new RomTensor({ 13,23,16 }, flt, s_ref_mul_in1_07);

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

  for(int i = 0; i < 4784; i++) {
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
  localCircularArenaAllocator<3312*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 16,23,9 }, flt, s_ref_div_out_07);
  Tensor out = new RamTensor({ 16,23,9 }, flt);
  Tensor in1 = new RomTensor({ 16,23,9 }, flt, s_ref_div_in1_07);
  Tensor in2 = new RomTensor({ 9 }, flt, s_ref_div_in2_07);

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

  for(int i = 0; i < 3312; i++) {
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
  localCircularArenaAllocator<228*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 4,1 }, flt, s_ref_sub_in2_08);
  Tensor ref_out = new RomTensor({ 19,4,3 }, flt, s_ref_sub_out_08);
  Tensor in1 = new RomTensor({ 19,4,3 }, flt, s_ref_sub_in1_08);
  Tensor out = new RamTensor({ 19,4,3 }, flt);

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

  for(int i = 0; i < 228; i++) {
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
  localCircularArenaAllocator<3520*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 11,10,16,2 }, flt, s_ref_add_in1_08);
  Tensor ref_out = new RomTensor({ 11,10,16,2 }, flt, s_ref_add_out_08);
  Tensor in2 = new RomTensor({ 2 }, flt, s_ref_add_in2_08);
  Tensor out = new RamTensor({ 11,10,16,2 }, flt);

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

  for(int i = 0; i < 3520; i++) {
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
  localCircularArenaAllocator<640*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 10,4,16 }, flt, s_ref_mul_in1_08);
  Tensor in2 = new RomTensor({ 16 }, flt, s_ref_mul_in2_08);
  Tensor ref_out = new RomTensor({ 10,4,16 }, flt, s_ref_mul_out_08);
  Tensor out = new RamTensor({ 10,4,16 }, flt);

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

  for(int i = 0; i < 640; i++) {
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
  localCircularArenaAllocator<23940*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 12,19,7,15 }, flt);
  Tensor ref_out = new RomTensor({ 12,19,7,15 }, flt, s_ref_div_out_08);
  Tensor in1 = new RomTensor({ 12,19,7,15 }, flt, s_ref_div_in1_08);
  Tensor in2 = new RomTensor({ 1,15 }, flt, s_ref_div_in2_08);

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

  for(int i = 0; i < 23940; i++) {
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
  localCircularArenaAllocator<968*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 22,11,4 }, flt);
  Tensor in1 = new RomTensor({ 22,11,4 }, flt, s_ref_sub_in1_09);
  Tensor in2 = new RomTensor({ 4 }, flt, s_ref_sub_in2_09);
  Tensor ref_out = new RomTensor({ 22,11,4 }, flt, s_ref_sub_out_09);

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

  for(int i = 0; i < 968; i++) {
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
  localCircularArenaAllocator<1000*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 4 }, flt, s_ref_add_in2_09);
  Tensor out = new RamTensor({ 10,25,4 }, flt);
  Tensor in1 = new RomTensor({ 10,25,4 }, flt, s_ref_add_in1_09);
  Tensor ref_out = new RomTensor({ 10,25,4 }, flt, s_ref_add_out_09);

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

  for(int i = 0; i < 1000; i++) {
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
  localCircularArenaAllocator<960*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 2,8,6,10 }, flt);
  Tensor in2 = new RomTensor({ 1,10 }, flt, s_ref_mul_in2_09);
  Tensor in1 = new RomTensor({ 2,8,6,10 }, flt, s_ref_mul_in1_09);
  Tensor ref_out = new RomTensor({ 2,8,6,10 }, flt, s_ref_mul_out_09);

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

  for(int i = 0; i < 960; i++) {
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
  localCircularArenaAllocator<756*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 9,2,3,14 }, flt, s_ref_div_in1_09);
  Tensor out = new RamTensor({ 9,2,3,14 }, flt);
  Tensor ref_out = new RomTensor({ 9,2,3,14 }, flt, s_ref_div_out_09);
  Tensor in2 = new RomTensor({ 1,3,1 }, flt, s_ref_div_in2_09);

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

  for(int i = 0; i < 756; i++) {
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
  localCircularArenaAllocator<3059*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 19,7,23 }, flt, s_ref_sub_in1_10);
  Tensor out = new RamTensor({ 19,7,23 }, flt);
  Tensor in2 = new RomTensor({ 7,23 }, flt, s_ref_sub_in2_10);
  Tensor ref_out = new RomTensor({ 19,7,23 }, flt, s_ref_sub_out_10);

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

  for(int i = 0; i < 3059; i++) {
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
  localCircularArenaAllocator<4250*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 17,10,25 }, flt, s_ref_add_in1_10);
  Tensor ref_out = new RomTensor({ 17,10,25 }, flt, s_ref_add_out_10);
  Tensor out = new RamTensor({ 17,10,25 }, flt);
  Tensor in2 = new RomTensor({ 1,25 }, flt, s_ref_add_in2_10);

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

  for(int i = 0; i < 4250; i++) {
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
  localCircularArenaAllocator<192*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 2,6,16 }, flt, s_ref_mul_in1_10);
  Tensor in2 = new RomTensor({ 6,16 }, flt, s_ref_mul_in2_10);
  Tensor ref_out = new RomTensor({ 2,6,16 }, flt, s_ref_mul_out_10);
  Tensor out = new RamTensor({ 2,6,16 }, flt);

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

  for(int i = 0; i < 192; i++) {
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
  localCircularArenaAllocator<5175*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 15,23,15 }, flt, s_ref_div_in1_10);
  Tensor out = new RamTensor({ 15,23,15 }, flt);
  Tensor ref_out = new RomTensor({ 15,23,15 }, flt, s_ref_div_out_10);
  Tensor in2 = new RomTensor({ 23,15 }, flt, s_ref_div_in2_10);

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

  for(int i = 0; i < 5175; i++) {
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
  localCircularArenaAllocator<54648*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 23,9,22,12 }, flt);
  Tensor in1 = new RomTensor({ 23,9,22,12 }, flt, s_ref_sub_in1_11);
  Tensor in2 = new RomTensor({ 12 }, flt, s_ref_sub_in2_11);
  Tensor ref_out = new RomTensor({ 23,9,22,12 }, flt, s_ref_sub_out_11);

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

  for(int i = 0; i < 54648; i++) {
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
  localCircularArenaAllocator<1500*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_add_in2_11);
  Tensor out = new RamTensor({ 10,10,15 }, flt);
  Tensor in1 = new RomTensor({ 10,10,15 }, flt, s_ref_add_in1_11);
  Tensor ref_out = new RomTensor({ 10,10,15 }, flt, s_ref_add_out_11);

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

  for(int i = 0; i < 1500; i++) {
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
  localCircularArenaAllocator<1848*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 22,6,14 }, flt);
  Tensor in2 = new RomTensor({ 6,1 }, flt, s_ref_mul_in2_11);
  Tensor in1 = new RomTensor({ 22,6,14 }, flt, s_ref_mul_in1_11);
  Tensor ref_out = new RomTensor({ 22,6,14 }, flt, s_ref_mul_out_11);

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

  for(int i = 0; i < 1848; i++) {
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
  localCircularArenaAllocator<96*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 4,3,8 }, flt, s_ref_div_in1_11);
  Tensor out = new RamTensor({ 4,3,8 }, flt);
  Tensor ref_out = new RomTensor({ 4,3,8 }, flt, s_ref_div_out_11);
  Tensor in2 = new RomTensor({ 8 }, flt, s_ref_div_in2_11);

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

  for(int i = 0; i < 96; i++) {
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
  localCircularArenaAllocator<22848*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 17 }, flt, s_ref_sub_in2_12);
  Tensor ref_out = new RomTensor({ 4,14,24,17 }, flt, s_ref_sub_out_12);
  Tensor in1 = new RomTensor({ 4,14,24,17 }, flt, s_ref_sub_in1_12);
  Tensor out = new RamTensor({ 4,14,24,17 }, flt);

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

  for(int i = 0; i < 22848; i++) {
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
  localCircularArenaAllocator<9504*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 22,18,24 }, flt, s_ref_add_out_12);
  Tensor in2 = new RomTensor({ 24 }, flt, s_ref_add_in2_12);
  Tensor in1 = new RomTensor({ 22,18,24 }, flt, s_ref_add_in1_12);
  Tensor out = new RamTensor({ 22,18,24 }, flt);

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

  for(int i = 0; i < 9504; i++) {
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
  localCircularArenaAllocator<5985*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 21,19,15 }, flt);
  Tensor ref_out = new RomTensor({ 21,19,15 }, flt, s_ref_mul_out_12);
  Tensor in1 = new RomTensor({ 21,19,15 }, flt, s_ref_mul_in1_12);
  Tensor in2 = new RomTensor({ 19,15 }, flt, s_ref_mul_in2_12);

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

  for(int i = 0; i < 5985; i++) {
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
  localCircularArenaAllocator<1584*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 22,12,6 }, flt);
  Tensor in2 = new RomTensor({ 12,6 }, flt, s_ref_div_in2_12);
  Tensor ref_out = new RomTensor({ 22,12,6 }, flt, s_ref_div_out_12);
  Tensor in1 = new RomTensor({ 22,12,6 }, flt, s_ref_div_in1_12);

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

  for(int i = 0; i < 1584; i++) {
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
  localCircularArenaAllocator<330*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 10,11,3 }, flt, s_ref_sub_out_13);
  Tensor in1 = new RomTensor({ 10,11,3 }, flt, s_ref_sub_in1_13);
  Tensor in2 = new RomTensor({ 3 }, flt, s_ref_sub_in2_13);
  Tensor out = new RamTensor({ 10,11,3 }, flt);

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

  for(int i = 0; i < 330; i++) {
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
  localCircularArenaAllocator<8976*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 22,17,24 }, flt);
  Tensor in1 = new RomTensor({ 22,17,24 }, flt, s_ref_add_in1_13);
  Tensor ref_out = new RomTensor({ 22,17,24 }, flt, s_ref_add_out_13);
  Tensor in2 = new RomTensor({ 17,24 }, flt, s_ref_add_in2_13);

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

  for(int i = 0; i < 8976; i++) {
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
  localCircularArenaAllocator<1200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 12,20,5 }, flt, s_ref_mul_in1_13);
  Tensor out = new RamTensor({ 12,20,5 }, flt);
  Tensor ref_out = new RomTensor({ 12,20,5 }, flt, s_ref_mul_out_13);
  Tensor in2 = new RomTensor({ 1,5 }, flt, s_ref_mul_in2_13);

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
 * Generated Test 56
 ***************************************/
TEST(ReferenceDiv, random_gen_div__13) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5600*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 4,8,25,7 }, flt, s_ref_div_in1_13);
  Tensor out = new RamTensor({ 4,8,25,7 }, flt);
  Tensor ref_out = new RomTensor({ 4,8,25,7 }, flt, s_ref_div_out_13);
  Tensor in2 = new RomTensor({ 1,1,7 }, flt, s_ref_div_in2_13);

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

  for(int i = 0; i < 5600; i++) {
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
  localCircularArenaAllocator<1944*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 12,18 }, flt, s_ref_sub_in2_14);
  Tensor out = new RamTensor({ 9,12,18 }, flt);
  Tensor ref_out = new RomTensor({ 9,12,18 }, flt, s_ref_sub_out_14);
  Tensor in1 = new RomTensor({ 9,12,18 }, flt, s_ref_sub_in1_14);

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

  for(int i = 0; i < 1944; i++) {
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
  localCircularArenaAllocator<11040*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 4,6,20,23 }, flt, s_ref_add_out_14);
  Tensor in1 = new RomTensor({ 4,6,20,23 }, flt, s_ref_add_in1_14);
  Tensor in2 = new RomTensor({ 20,23 }, flt, s_ref_add_in2_14);
  Tensor out = new RamTensor({ 4,6,20,23 }, flt);

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

  for(int i = 0; i < 11040; i++) {
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
  localCircularArenaAllocator<270*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 3,10,9 }, flt, s_ref_mul_in1_14);
  Tensor out = new RamTensor({ 3,10,9 }, flt);
  Tensor in2 = new RomTensor({ 9 }, flt, s_ref_mul_in2_14);
  Tensor ref_out = new RomTensor({ 3,10,9 }, flt, s_ref_mul_out_14);

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

  for(int i = 0; i < 270; i++) {
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
  localCircularArenaAllocator<7980*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 20,21,19 }, flt, s_ref_div_out_14);
  Tensor in2 = new RomTensor({ 19 }, flt, s_ref_div_in2_14);
  Tensor in1 = new RomTensor({ 20,21,19 }, flt, s_ref_div_in1_14);
  Tensor out = new RamTensor({ 20,21,19 }, flt);

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

  for(int i = 0; i < 7980; i++) {
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
  localCircularArenaAllocator<1890*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 9,21,10 }, flt, s_ref_sub_out_15);
  Tensor in2 = new RomTensor({ 21,1 }, flt, s_ref_sub_in2_15);
  Tensor out = new RamTensor({ 9,21,10 }, flt);
  Tensor in1 = new RomTensor({ 9,21,10 }, flt, s_ref_sub_in1_15);

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

  for(int i = 0; i < 1890; i++) {
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
  localCircularArenaAllocator<10080*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 14,18,4,10 }, flt, s_ref_add_out_15);
  Tensor in1 = new RomTensor({ 14,18,4,10 }, flt, s_ref_add_in1_15);
  Tensor in2 = new RomTensor({ 10 }, flt, s_ref_add_in2_15);
  Tensor out = new RamTensor({ 14,18,4,10 }, flt);

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

  for(int i = 0; i < 10080; i++) {
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
  localCircularArenaAllocator<9200*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 25,16,23 }, flt, s_ref_mul_in1_15);
  Tensor ref_out = new RomTensor({ 25,16,23 }, flt, s_ref_mul_out_15);
  Tensor in2 = new RomTensor({ 1,23 }, flt, s_ref_mul_in2_15);
  Tensor out = new RamTensor({ 25,16,23 }, flt);

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

  for(int i = 0; i < 9200; i++) {
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
  localCircularArenaAllocator<60192*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 16,11,18,19 }, flt, s_ref_div_out_15);
  Tensor in1 = new RomTensor({ 16,11,18,19 }, flt, s_ref_div_in1_15);
  Tensor out = new RamTensor({ 16,11,18,19 }, flt);
  Tensor in2 = new RomTensor({ 1,19 }, flt, s_ref_div_in2_15);

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

  for(int i = 0; i < 60192; i++) {
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
  localCircularArenaAllocator<7560*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 14,15,18,2 }, flt, s_ref_sub_out_16);
  Tensor in2 = new RomTensor({ 18,2 }, flt, s_ref_sub_in2_16);
  Tensor out = new RamTensor({ 14,15,18,2 }, flt);
  Tensor in1 = new RomTensor({ 14,15,18,2 }, flt, s_ref_sub_in1_16);

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

  for(int i = 0; i < 7560; i++) {
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
  localCircularArenaAllocator<1482*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 13,19 }, flt, s_ref_add_in2_16);
  Tensor out = new RamTensor({ 3,2,13,19 }, flt);
  Tensor in1 = new RomTensor({ 3,2,13,19 }, flt, s_ref_add_in1_16);
  Tensor ref_out = new RomTensor({ 3,2,13,19 }, flt, s_ref_add_out_16);

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

  for(int i = 0; i < 1482; i++) {
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
  localCircularArenaAllocator<1232*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 8 }, flt, s_ref_mul_in2_16);
  Tensor ref_out = new RomTensor({ 22,7,8 }, flt, s_ref_mul_out_16);
  Tensor in1 = new RomTensor({ 22,7,8 }, flt, s_ref_mul_in1_16);
  Tensor out = new RamTensor({ 22,7,8 }, flt);

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

  for(int i = 0; i < 1232; i++) {
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
  localCircularArenaAllocator<1350*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 3,6,25,3 }, flt, s_ref_div_in1_16);
  Tensor in2 = new RomTensor({ 3 }, flt, s_ref_div_in2_16);
  Tensor out = new RamTensor({ 3,6,25,3 }, flt);
  Tensor ref_out = new RomTensor({ 3,6,25,3 }, flt, s_ref_div_out_16);

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

  for(int i = 0; i < 1350; i++) {
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
  localCircularArenaAllocator<96*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 16 }, flt, s_ref_sub_in2_17);
  Tensor out = new RamTensor({ 2,3,16 }, flt);
  Tensor in1 = new RomTensor({ 2,3,16 }, flt, s_ref_sub_in1_17);
  Tensor ref_out = new RomTensor({ 2,3,16 }, flt, s_ref_sub_out_17);

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

  for(int i = 0; i < 96; i++) {
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
  localCircularArenaAllocator<330*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 10,1 }, flt, s_ref_add_in2_17);
  Tensor out = new RamTensor({ 11,10,3 }, flt);
  Tensor in1 = new RomTensor({ 11,10,3 }, flt, s_ref_add_in1_17);
  Tensor ref_out = new RomTensor({ 11,10,3 }, flt, s_ref_add_out_17);

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

  for(int i = 0; i < 330; i++) {
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
  localCircularArenaAllocator<32490*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 6,19,19,15 }, flt);
  Tensor in1 = new RomTensor({ 6,19,19,15 }, flt, s_ref_mul_in1_17);
  Tensor ref_out = new RomTensor({ 6,19,19,15 }, flt, s_ref_mul_out_17);
  Tensor in2 = new RomTensor({ 15 }, flt, s_ref_mul_in2_17);

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

  for(int i = 0; i < 32490; i++) {
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
  localCircularArenaAllocator<16896*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in2 = new RomTensor({ 22,2,16 }, flt, s_ref_div_in2_17);
  Tensor in1 = new RomTensor({ 24,22,2,16 }, flt, s_ref_div_in1_17);
  Tensor out = new RamTensor({ 24,22,2,16 }, flt);
  Tensor ref_out = new RomTensor({ 24,22,2,16 }, flt, s_ref_div_out_17);

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

  for(int i = 0; i < 16896; i++) {
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
  localCircularArenaAllocator<975*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 15,5,13 }, flt, s_ref_sub_in1_18);
  Tensor out = new RamTensor({ 15,5,13 }, flt);
  Tensor in2 = new RomTensor({ 5,1 }, flt, s_ref_sub_in2_18);
  Tensor ref_out = new RomTensor({ 15,5,13 }, flt, s_ref_sub_out_18);

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

  for(int i = 0; i < 975; i++) {
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
  localCircularArenaAllocator<2156*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 14,14,11 }, flt);
  Tensor in2 = new RomTensor({ 11 }, flt, s_ref_add_in2_18);
  Tensor in1 = new RomTensor({ 14,14,11 }, flt, s_ref_add_in1_18);
  Tensor ref_out = new RomTensor({ 14,14,11 }, flt, s_ref_add_out_18);

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

  for(int i = 0; i < 2156; i++) {
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
  localCircularArenaAllocator<9500*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 25,20,19 }, flt, s_ref_mul_in1_18);
  Tensor out = new RamTensor({ 25,20,19 }, flt);
  Tensor in2 = new RomTensor({ 19 }, flt, s_ref_mul_in2_18);
  Tensor ref_out = new RomTensor({ 25,20,19 }, flt, s_ref_mul_out_18);

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

  for(int i = 0; i < 9500; i++) {
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
  localCircularArenaAllocator<103680*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 15,24,18,16 }, flt, s_ref_div_out_18);
  Tensor out = new RamTensor({ 15,24,18,16 }, flt);
  Tensor in1 = new RomTensor({ 15,24,18,16 }, flt, s_ref_div_in1_18);
  Tensor in2 = new RomTensor({ 24,1,16 }, flt, s_ref_div_in2_18);

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

  for(int i = 0; i < 103680; i++) {
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
  localCircularArenaAllocator<137088*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 24,17,24,14 }, flt);
  Tensor in2 = new RomTensor({ 17,24,14 }, flt, s_ref_sub_in2_19);
  Tensor ref_out = new RomTensor({ 24,17,24,14 }, flt, s_ref_sub_out_19);
  Tensor in1 = new RomTensor({ 24,17,24,14 }, flt, s_ref_sub_in1_19);

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

  for(int i = 0; i < 137088; i++) {
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
  localCircularArenaAllocator<3872*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 11,22,16 }, flt, s_ref_add_out_19);
  Tensor out = new RamTensor({ 11,22,16 }, flt);
  Tensor in2 = new RomTensor({ 22,16 }, flt, s_ref_add_in2_19);
  Tensor in1 = new RomTensor({ 11,22,16 }, flt, s_ref_add_in1_19);

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

  for(int i = 0; i < 3872; i++) {
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
  localCircularArenaAllocator<2288*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor in1 = new RomTensor({ 22,8,13 }, flt, s_ref_mul_in1_19);
  Tensor ref_out = new RomTensor({ 22,8,13 }, flt, s_ref_mul_out_19);
  Tensor out = new RamTensor({ 22,8,13 }, flt);
  Tensor in2 = new RomTensor({ 8,13 }, flt, s_ref_mul_in2_19);

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

  for(int i = 0; i < 2288; i++) {
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
  localCircularArenaAllocator<980*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 5,14,14 }, flt, s_ref_div_out_19);
  Tensor in1 = new RomTensor({ 5,14,14 }, flt, s_ref_div_in1_19);
  Tensor out = new RamTensor({ 5,14,14 }, flt);
  Tensor in2 = new RomTensor({ 14 }, flt, s_ref_div_in2_19);

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

  for(int i = 0; i < 980; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-06);
}

}

