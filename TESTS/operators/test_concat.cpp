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
  localCircularArenaAllocator<4620*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_00);
  Tensor b = new RomTensor({ 9,6,7,5 }, flt, ref_in_b_00);
  Tensor a = new RomTensor({ 13,6,7,5 }, flt, ref_in_a_00);
  Tensor out = new RamTensor({ 22,6,7,5 }, flt);
  Tensor ref_out = new RomTensor({ 22,6,7,5 }, flt, ref_out_00);

  ReferenceOperators::ConcatOperator concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator::a, a },
    { ReferenceOperators::ConcatOperator::b, b },
    { ReferenceOperators::ConcatOperator::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4620; i++) {
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
  localCircularArenaAllocator<13*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b = new RomTensor({ 4 }, flt, ref_in_b_01);
  Tensor ref_out = new RomTensor({ 13 }, flt, ref_out_01);
  Tensor out = new RamTensor({ 13 }, flt);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_01);
  Tensor a = new RomTensor({ 9 }, flt, ref_in_a_01);

  ReferenceOperators::ConcatOperator concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator::a, a },
    { ReferenceOperators::ConcatOperator::b, b },
    { ReferenceOperators::ConcatOperator::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 13; i++) {
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
  localCircularArenaAllocator<240*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_02);
  Tensor b = new RomTensor({ 4,10,3 }, flt, ref_in_b_02);
  Tensor a = new RomTensor({ 4,10,3 }, flt, ref_in_a_02);
  Tensor ref_out = new RomTensor({ 4,20,3 }, flt, ref_out_02);
  Tensor out = new RamTensor({ 4,20,3 }, flt);

  ReferenceOperators::ConcatOperator concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator::a, a },
    { ReferenceOperators::ConcatOperator::b, b },
    { ReferenceOperators::ConcatOperator::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 240; i++) {
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
  localCircularArenaAllocator<27*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 1,3 }, flt, ref_in_a_03);
  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_03);
  Tensor out = new RamTensor({ 9,3 }, flt);
  Tensor b = new RomTensor({ 8,3 }, flt, ref_in_b_03);
  Tensor ref_out = new RomTensor({ 9,3 }, flt, ref_out_03);

  ReferenceOperators::ConcatOperator concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator::a, a },
    { ReferenceOperators::ConcatOperator::b, b },
    { ReferenceOperators::ConcatOperator::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 27; i++) {
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
  localCircularArenaAllocator<190*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor axis = new RomTensor({ 1 }, i32, ref_axis_04);
  Tensor b = new RomTensor({ 10,7 }, flt, ref_in_b_04);
  Tensor ref_out = new RomTensor({ 10,19 }, flt, ref_out_04);
  Tensor a = new RomTensor({ 10,12 }, flt, ref_in_a_04);
  Tensor out = new RamTensor({ 10,19 }, flt);

  ReferenceOperators::ConcatOperator concat_op;
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator::a, a },
    { ReferenceOperators::ConcatOperator::b, b },
    { ReferenceOperators::ConcatOperator::axis, axis }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 190; i++) {
  EXPECT_NEAR(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ), 1e-07);
}
}

