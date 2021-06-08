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
 * Generated Test 1
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<567*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 9,9,7 }, flt);
  Tensor a = new RomTensor({ 9,3,7 }, flt, ref_in_a_00);
  Tensor ref_out = new RomTensor({ 9,9,7 }, flt, ref_out_00);
  Tensor b = new RomTensor({ 9,6,7 }, flt, ref_in_b_00);

  ReferenceOperators::ConcatOperator<float> concat_op(-2);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 567; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__01) {
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

  Tensor a = new RomTensor({ 6,6 }, flt, ref_in_a_01);
  Tensor out = new RamTensor({ 10,6 }, flt);
  Tensor ref_out = new RomTensor({ 10,6 }, flt, ref_out_01);
  Tensor b = new RomTensor({ 4,6 }, flt, ref_in_b_01);

  ReferenceOperators::ConcatOperator<float> concat_op(0);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 60; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<294*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b = new RomTensor({ 14,1,13 }, flt, ref_in_b_02);
  Tensor ref_out = new RomTensor({ 14,1,21 }, flt, ref_out_02);
  Tensor a = new RomTensor({ 14,1,8 }, flt, ref_in_a_02);
  Tensor out = new RamTensor({ 14,1,21 }, flt);

  ReferenceOperators::ConcatOperator<float> concat_op(-1);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 294; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5775*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 7,2,5,11 }, flt, ref_in_a_03);
  Tensor ref_out = new RomTensor({ 7,15,5,11 }, flt, ref_out_03);
  Tensor b = new RomTensor({ 7,13,5,11 }, flt, ref_in_b_03);
  Tensor out = new RamTensor({ 7,15,5,11 }, flt);

  ReferenceOperators::ConcatOperator<float> concat_op(1);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5775; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__04) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<1960*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor a = new RomTensor({ 7,2,14,7 }, flt, ref_in_a_04);
  Tensor out = new RamTensor({ 10,2,14,7 }, flt);
  Tensor b = new RomTensor({ 3,2,14,7 }, flt, ref_in_b_04);
  Tensor ref_out = new RomTensor({ 10,2,14,7 }, flt, ref_out_04);

  ReferenceOperators::ConcatOperator<float> concat_op(0);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1960; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__05) {
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

  Tensor a = new RomTensor({ 5,2 }, flt, ref_in_a_05);
  Tensor out = new RamTensor({ 10,2 }, flt);
  Tensor ref_out = new RomTensor({ 10,2 }, flt, ref_out_05);
  Tensor b = new RomTensor({ 5,2 }, flt, ref_in_b_05);

  ReferenceOperators::ConcatOperator<float> concat_op(-2);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 20; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 7
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<720*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor b = new RomTensor({ 5,10,4 }, flt, ref_in_b_06);
  Tensor out = new RamTensor({ 18,10,4 }, flt);
  Tensor a = new RomTensor({ 13,10,4 }, flt, ref_in_a_06);
  Tensor ref_out = new RomTensor({ 18,10,4 }, flt, ref_out_06);

  ReferenceOperators::ConcatOperator<float> concat_op(0);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 720; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 8
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__07) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<792*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 3,6,4,11 }, flt);
  Tensor ref_out = new RomTensor({ 3,6,4,11 }, flt, ref_out_07);
  Tensor b = new RomTensor({ 3,6,4,9 }, flt, ref_in_b_07);
  Tensor a = new RomTensor({ 3,6,4,2 }, flt, ref_in_a_07);

  ReferenceOperators::ConcatOperator<float> concat_op(-1);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 792; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 9
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__08) {
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

  Tensor b = new RomTensor({ 1 }, flt, ref_in_b_08);
  Tensor ref_out = new RomTensor({ 3 }, flt, ref_out_08);
  Tensor a = new RomTensor({ 2 }, flt, ref_in_a_08);
  Tensor out = new RamTensor({ 3 }, flt);

  ReferenceOperators::ConcatOperator<float> concat_op(-1);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 3; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

/***************************************
 * Generated Test 10
 ***************************************/
TEST(ReferenceConcat, random_gen_concat__09) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<210*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor ref_out = new RomTensor({ 14,15 }, flt, ref_out_09);
  Tensor b = new RomTensor({ 14,7 }, flt, ref_in_b_09);
  Tensor out = new RamTensor({ 14,15 }, flt);
  Tensor a = new RomTensor({ 14,8 }, flt, ref_in_a_09);

  ReferenceOperators::ConcatOperator<float> concat_op(-1);
  concat_op
  .set_inputs({ 
    { ReferenceOperators::ConcatOperator<float>::a, a },
    { ReferenceOperators::ConcatOperator<float>::b, b }
  }).set_outputs({ 
    { ReferenceOperators::ConcatOperator<float>::out, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 210; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( ref_out(i) ));
}

}

