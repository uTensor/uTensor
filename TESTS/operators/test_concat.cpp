#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_concat.hpp"
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
  localCircularArenaAllocator<160*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 8,20 }, flt, ref_out_00);
  Tensor b = new RomTensor({ 8,13 }, flt, ref_in_b_00);
  Tensor out = new RamTensor({ 8,20 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_00);
  Tensor a = new RomTensor({ 8,7 }, flt, ref_in_a_00);

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

  for(int i = 0; i < 160; i++) {
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
  localCircularArenaAllocator<11*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 11,1 }, flt);
  Tensor a = new RomTensor({ 10,1 }, flt, ref_in_a_01);
  Tensor ref_out = new RomTensor({ 11,1 }, flt, ref_out_01);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_01);
  Tensor b = new RomTensor({ 1,1 }, flt, ref_in_b_01);

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

  for(int i = 0; i < 11; i++) {
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
  localCircularArenaAllocator<4368*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b = new RomTensor({ 12,13,14 }, flt, ref_in_b_02);
  Tensor out = new RamTensor({ 12,13,28 }, flt);
  Tensor a = new RomTensor({ 12,13,14 }, flt, ref_in_a_02);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_02);
  Tensor ref_out = new RomTensor({ 12,13,28 }, flt, ref_out_02);

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

  for(int i = 0; i < 4368; i++) {
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
  localCircularArenaAllocator<195*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 13,15 }, flt, ref_out_03);
  Tensor out = new RamTensor({ 13,15 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_03);
  Tensor b = new RomTensor({ 13,2 }, flt, ref_in_b_03);
  Tensor a = new RomTensor({ 13,13 }, flt, ref_in_a_03);

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

  for(int i = 0; i < 195; i++) {
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
  localCircularArenaAllocator<4004*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_04);
  Tensor b = new RomTensor({ 11,14,2,2 }, flt, ref_in_b_04);
  Tensor a = new RomTensor({ 11,14,11,2 }, flt, ref_in_a_04);
  Tensor ref_out = new RomTensor({ 11,14,13,2 }, flt, ref_out_04);
  Tensor out = new RamTensor({ 11,14,13,2 }, flt);

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

  for(int i = 0; i < 4004; i++) {
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
  localCircularArenaAllocator<175*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_05);
  Tensor a = new RomTensor({ 12,7 }, flt, ref_in_a_05);
  Tensor out = new RamTensor({ 25,7 }, flt);
  Tensor ref_out = new RomTensor({ 25,7 }, flt, ref_out_05);
  Tensor b = new RomTensor({ 13,7 }, flt, ref_in_b_05);

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

  for(int i = 0; i < 175; i++) {
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
  localCircularArenaAllocator<6*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 3 }, flt, ref_in_a_06);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_06);
  Tensor out = new RamTensor({ 6 }, flt);
  Tensor b = new RomTensor({ 3 }, flt, ref_in_b_06);
  Tensor ref_out = new RomTensor({ 6 }, flt, ref_out_06);

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

  for(int i = 0; i < 6; i++) {
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
  localCircularArenaAllocator<56*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 4,1,8 }, flt, ref_in_a_07);
  Tensor ref_out = new RomTensor({ 4,1,14 }, flt, ref_out_07);
  Tensor b = new RomTensor({ 4,1,6 }, flt, ref_in_b_07);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_07);
  Tensor out = new RamTensor({ 4,1,14 }, flt);

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

  for(int i = 0; i < 56; i++) {
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
  localCircularArenaAllocator<110*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_08);
  Tensor ref_out = new RomTensor({ 11,10,1 }, flt, ref_out_08);
  Tensor b = new RomTensor({ 4,10,1 }, flt, ref_in_b_08);
  Tensor out = new RamTensor({ 11,10,1 }, flt);
  Tensor a = new RomTensor({ 7,10,1 }, flt, ref_in_a_08);

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

  for(int i = 0; i < 110; i++) {
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
  localCircularArenaAllocator<24*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 2,12 }, flt, ref_out_09);
  Tensor out = new RamTensor({ 2,12 }, flt);
  Tensor a = new RomTensor({ 2,4 }, flt, ref_in_a_09);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_09);
  Tensor b = new RomTensor({ 2,8 }, flt, ref_in_b_09);

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

  for(int i = 0; i < 24; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

