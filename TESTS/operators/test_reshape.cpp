#include <cstring>
#include <iostream>

#include "uTensor.h"
#include "gtest/gtest.h"

#include "constants_reshape.hpp"
using std::cout;
using std::endl;

using namespace uTensor;

SimpleErrorHandler mErrHandler(10);

/***************************************
 * Generated Test 1
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__00) {
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

  Tensor in = new RomTensor({ 3,5,4 }, flt, s_ref_in_00);
  Tensor out_ref = new RomTensor({ 2,2,3,5 }, flt, s_ref_out_00);
  Tensor out = new RamTensor({ 2,2,3,5 }, flt);

  ReferenceOperators::ReshapeOperator<float> reshape_op({2, 2, 3, 5});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 60; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

/***************************************
 * Generated Test 2
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__01) {
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

  Tensor out_ref = new RomTensor({ 2,5,3 }, flt, s_ref_out_01);
  Tensor out = new RamTensor({ 2,5,3 }, flt);
  Tensor in = new RomTensor({ 10,3 }, flt, s_ref_in_01);

  ReferenceOperators::ReshapeOperator<float> reshape_op({2, 5, 3});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 30; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

/***************************************
 * Generated Test 3
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__02) {
  // Make sure no errors get thrown
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err){
      got_error = true;
  });

  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<5*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);

  Tensor out = new RamTensor({ 1,1,5 }, flt);
  Tensor out_ref = new RomTensor({ 1,1,5 }, flt, s_ref_out_02);
  Tensor in = new RomTensor({ 1,5 }, flt, s_ref_in_02);

  ReferenceOperators::ReshapeOperator<float> reshape_op({1, 1, 5});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 5; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

/***************************************
 * Generated Test 4
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__03) {
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

  Tensor in = new RomTensor({ 3,5 }, flt, s_ref_in_03);
  Tensor out = new RamTensor({ 1,3,5 }, flt);
  Tensor out_ref = new RomTensor({ 1,3,5 }, flt, s_ref_out_03);

  ReferenceOperators::ReshapeOperator<float> reshape_op({1, 3, 5});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 15; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

/***************************************
 * Generated Test 5
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__04) {
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

  Tensor out_ref = new RomTensor({ 3,1,5 }, flt, s_ref_out_04);
  Tensor in = new RomTensor({ 3,5 }, flt, s_ref_in_04);
  Tensor out = new RamTensor({ 3,1,5 }, flt);

  ReferenceOperators::ReshapeOperator<float> reshape_op({3, 1, 5});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 15; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

/***************************************
 * Generated Test 6
 ***************************************/
TEST(ReferenceReshape, random_gen_reshape__05) {
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

  Tensor in = new RomTensor({ 3,5 }, flt, s_ref_in_05);
  Tensor out = new RamTensor({ 3,5,1 }, flt);
  Tensor out_ref = new RomTensor({ 3,5,1 }, flt, s_ref_out_05);

  ReferenceOperators::ReshapeOperator<float> reshape_op({3, 5, 1});
  reshape_op
  .set_inputs({ 
    { ReferenceOperators::ReshapeOperator<float>::input, in }
  }).set_outputs({ 
    { ReferenceOperators::ReshapeOperator<float>::output, out }
  }).eval();

  // Make sure no errors got thrown
  ASSERT_EQ(got_error, false);

  for(int i = 0; i < 15; i++) {
  EXPECT_EQ(static_cast<float>( out(i) ), static_cast<float>( out_ref(i) ));
}

}

