#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_stride_slice.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test 1
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__00) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor tensor_end = new RomTensor({ 1 }, i32, ref_end_00);
  Tensor tensor_begin = new RomTensor({ 1 }, i32, ref_begin_00);
  Tensor ref_tensor_output = new RomTensor({ 4 }, flt, ref_out_00);
  Tensor x = new RomTensor({ 5 }, flt, ref_in_x_00);
  Tensor tensor_output = new RamTensor({ 4 }, flt);
  Tensor tensor_strides = new RomTensor({ 1 }, i32, ref_strides_00);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(1, 0, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__01) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor tensor_output = new RamTensor({ 4,1,1,1 }, flt);
  Tensor x = new RomTensor({ 4,2,5,3 }, flt, ref_in_x_01);
  Tensor tensor_strides = new RomTensor({ 4 }, i32, ref_strides_01);
  Tensor tensor_end = new RomTensor({ 4 }, i32, ref_end_01);
  Tensor ref_tensor_output = new RomTensor({ 4,1,1,1 }, flt, ref_out_01);
  Tensor tensor_begin = new RomTensor({ 4 }, i32, ref_begin_01);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(7, 1, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<48*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor x = new RomTensor({ 5,5,4,4 }, flt, ref_in_x_02);
  Tensor ref_tensor_output = new RomTensor({ 2,3,2,4 }, flt, ref_out_02);
  Tensor tensor_end = new RomTensor({ 4 }, i32, ref_end_02);
  Tensor tensor_output = new RamTensor({ 2,3,2,4 }, flt);
  Tensor tensor_begin = new RomTensor({ 4 }, i32, ref_begin_02);
  Tensor tensor_strides = new RomTensor({ 4 }, i32, ref_strides_02);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(15, 14, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 48; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__03) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<9*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor tensor_output = new RamTensor({ 3,3 }, flt);
  Tensor tensor_end = new RomTensor({ 2 }, i32, ref_end_03);
  Tensor tensor_begin = new RomTensor({ 2 }, i32, ref_begin_03);
  Tensor x = new RomTensor({ 6,6 }, flt, ref_in_x_03);
  Tensor ref_tensor_output = new RomTensor({ 3,3 }, flt, ref_out_03);
  Tensor tensor_strides = new RomTensor({ 2 }, i32, ref_strides_03);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(0, 2, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 9; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__04) {
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

  Tensor tensor_begin = new RomTensor({ 2 }, i32, ref_begin_04);
  Tensor tensor_output = new RamTensor({ 1,1 }, flt);
  Tensor x = new RomTensor({ 1,5 }, flt, ref_in_x_04);
  Tensor tensor_end = new RomTensor({ 2 }, i32, ref_end_04);
  Tensor tensor_strides = new RomTensor({ 2 }, i32, ref_strides_04);
  Tensor ref_tensor_output = new RomTensor({ 1,1 }, flt, ref_out_04);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(1, 1, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__05) {
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

  Tensor x = new RomTensor({ 4 }, flt, ref_in_x_05);
  Tensor tensor_strides = new RomTensor({ 1 }, i32, ref_strides_05);
  Tensor tensor_end = new RomTensor({ 1 }, i32, ref_end_05);
  Tensor tensor_begin = new RomTensor({ 1 }, i32, ref_begin_05);
  Tensor ref_tensor_output = new RomTensor({ 1 }, flt, ref_out_05);
  Tensor tensor_output = new RamTensor({ 1 }, flt);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(1, 0, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 7
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__06) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<4*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor tensor_end = new RomTensor({ 1 }, i32, ref_end_06);
  Tensor x = new RomTensor({ 4 }, flt, ref_in_x_06);
  Tensor ref_tensor_output = new RomTensor({ 4 }, flt, ref_out_06);
  Tensor tensor_output = new RamTensor({ 4 }, flt);
  Tensor tensor_strides = new RomTensor({ 1 }, i32, ref_strides_06);
  Tensor tensor_begin = new RomTensor({ 1 }, i32, ref_begin_06);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(0, 1, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 4; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 8
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__07) {
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

  Tensor ref_tensor_output = new RomTensor({ 6,3 }, flt, ref_out_07);
  Tensor x = new RomTensor({ 9,8 }, flt, ref_in_x_07);
  Tensor tensor_output = new RamTensor({ 6,3 }, flt);
  Tensor tensor_begin = new RomTensor({ 2 }, i32, ref_begin_07);
  Tensor tensor_strides = new RomTensor({ 2 }, i32, ref_strides_07);
  Tensor tensor_end = new RomTensor({ 2 }, i32, ref_end_07);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(2, 1, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 18; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 9
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__08) {
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

  Tensor ref_tensor_output = new RomTensor({ 3,8,6,2 }, flt, ref_out_08);
  Tensor tensor_begin = new RomTensor({ 4 }, i32, ref_begin_08);
  Tensor tensor_output = new RamTensor({ 3,8,6,2 }, flt);
  Tensor tensor_end = new RomTensor({ 4 }, i32, ref_end_08);
  Tensor x = new RomTensor({ 9,8,8,4 }, flt, ref_in_x_08);
  Tensor tensor_strides = new RomTensor({ 4 }, i32, ref_strides_08);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(0, 14, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 288; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

/***************************************
 * Generated Test 10
 ***************************************/
TEST(ReferenceStridedSlice, random_gen_strided_slice__09) {
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

  Tensor tensor_output = new RamTensor({ 1 }, flt);
  Tensor ref_tensor_output = new RomTensor({ 1 }, flt, ref_out_09);
  Tensor tensor_begin = new RomTensor({ 1 }, i32, ref_begin_09);
  Tensor tensor_end = new RomTensor({ 1 }, i32, ref_end_09);
  Tensor x = new RomTensor({ 7 }, flt, ref_in_x_09);
  Tensor tensor_strides = new RomTensor({ 1 }, i32, ref_strides_09);

  ReferenceOperators::StridedSliceOperator<float> strided_slice_op(1, 0, 0, 0, 0);
  strided_slice_op
  .set_inputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::input, x },
    { ReferenceOperators::StridedSliceOperator<float>::begin, tensor_begin },
    { ReferenceOperators::StridedSliceOperator<float>::end, tensor_end },
    { ReferenceOperators::StridedSliceOperator<float>::strides, tensor_strides }
  }).set_outputs({ 
    { ReferenceOperators::StridedSliceOperator<float>::output, tensor_output }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 1; i++) {
  EXPECT_NEAR(static_cast<float>( tensor_output(i) ), static_cast<float>( ref_tensor_output(i) ), 1e-07);
}
}

