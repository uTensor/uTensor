#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "/Users/dboyliao/Work/open_source/uTensor/uTensor/TESTS/operators/constants_concat.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__00) {
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

  Tensor b = new RomTensor({ 7 }, flt, ref_in_b_00);
  Tensor out = new RamTensor({ 12 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_00);
  Tensor ref_out = new RomTensor({ 12 }, flt, ref_out_00);
  Tensor a = new RomTensor({ 5 }, flt, ref_in_a_00);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 12; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__01) {
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

  Tensor ref_out = new RomTensor({ 8 }, flt, ref_out_01);
  Tensor b = new RomTensor({ 5 }, flt, ref_in_b_01);
  Tensor out = new RamTensor({ 8 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_01);
  Tensor a = new RomTensor({ 3 }, flt, ref_in_a_01);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 8; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<2496*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 13,24,8 }, flt);
  Tensor ref_out = new RomTensor({ 13,24,8 }, flt, ref_out_02);
  Tensor a = new RomTensor({ 13,12,8 }, flt, ref_in_a_02);
  Tensor b = new RomTensor({ 13,12,8 }, flt, ref_in_b_02);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_02);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 2496; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5880*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 6,6,14,5 }, flt, ref_in_a_03);
  Tensor b = new RomTensor({ 8,6,14,5 }, flt, ref_in_b_03);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_03);
  Tensor out = new RamTensor({ 14,6,14,5 }, flt);
  Tensor ref_out = new RomTensor({ 14,6,14,5 }, flt, ref_out_03);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5880; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__04) {
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

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_04);
  Tensor a = new RomTensor({ 9,3 }, flt, ref_in_a_04);
  Tensor ref_out = new RomTensor({ 9,6 }, flt, ref_out_04);
  Tensor out = new RamTensor({ 9,6 }, flt);
  Tensor b = new RomTensor({ 9,3 }, flt, ref_in_b_04);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 54; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__05) {
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

  Tensor out = new RamTensor({ 12 }, flt);
  Tensor b = new RomTensor({ 7 }, flt, ref_in_b_05);
  Tensor a = new RomTensor({ 5 }, flt, ref_in_a_05);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_05);
  Tensor ref_out = new RomTensor({ 12 }, flt, ref_out_05);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 12; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<13860*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 10,14,6,9 }, flt, ref_in_a_06);
  Tensor out = new RamTensor({ 10,14,11,9 }, flt);
  Tensor ref_out = new RomTensor({ 10,14,11,9 }, flt, ref_out_06);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_06);
  Tensor b = new RomTensor({ 10,14,5,9 }, flt, ref_in_b_06);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 13860; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__07) {
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

  Tensor out = new RamTensor({ 10,10,14 }, flt);
  Tensor ref_out = new RomTensor({ 10,10,14 }, flt, ref_out_07);
  Tensor b = new RomTensor({ 6,10,14 }, flt, ref_in_b_07);
  Tensor a = new RomTensor({ 4,10,14 }, flt, ref_in_a_07);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_07);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1400; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__08) {
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

  Tensor out = new RamTensor({ 13,13 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_08);
  Tensor b = new RomTensor({ 13,4 }, flt, ref_in_b_08);
  Tensor a = new RomTensor({ 13,9 }, flt, ref_in_a_08);
  Tensor ref_out = new RomTensor({ 13,13 }, flt, ref_out_08);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 169; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

/***************************************
 * Generated Test
 ***************************************/

TEST(ReferenceConcat, random_gen_concat__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1638*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 3,26,7,3 }, flt, ref_out_09);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_09);
  Tensor a = new RomTensor({ 3,12,7,3 }, flt, ref_in_a_09);
  Tensor out = new RamTensor({ 3,26,7,3 }, flt);
  Tensor b = new RomTensor({ 3,14,7,3 }, flt, ref_in_b_09);

  ReferenceOperators::ConcatOperator<float> concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b },
    { ReferenceOperators::ConcatOperator<float>::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1638; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

