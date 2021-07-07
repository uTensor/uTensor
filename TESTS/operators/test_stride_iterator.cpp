
#include "gtest/gtest.h"
#include "uTensor.h"

using namespace uTensor;



/*
  Random Generated Test Number 00
*/
TEST(StridedIterator, test_strided_it_00) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  int32_t s_begin[2] = { 7, 0 };
  int32_t s_end[2] = { 8, 1 };
  int32_t s_strides[2] = { 1, 1 };
  int32_t ref_li[1] = { 14 };

  Tensor input_tensor = new BufferTensor({ 8, 2 }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ 2 }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ 2 }, i32, s_end);
  int32_t begin_mask = 0;
  int32_t end_mask = 1;
  Tensor strides_tensor = new BufferTensor({ 2 }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), 1);
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}


/*
  Random Generated Test Number 01
*/
TEST(StridedIterator, test_strided_it_01) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[6] = { 0, 1, 2, 3, 4, 5 };
  int32_t s_begin[1] = { 0 };
  int32_t s_end[1] = { 5 };
  int32_t s_strides[1] = { 2 };
  int32_t ref_li[3] = { 0, 2, 4 };

  Tensor input_tensor = new BufferTensor({ 6 }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ 1 }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ 1 }, i32, s_end);
  int32_t begin_mask = 1;
  int32_t end_mask = 0;
  Tensor strides_tensor = new BufferTensor({ 1 }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), 3);
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}


/*
  Random Generated Test Number 02
*/
TEST(StridedIterator, test_strided_it_02) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[7] = { 0, 1, 2, 3, 4, 5, 6 };
  int32_t s_begin[1] = { 5 };
  int32_t s_end[1] = { 6 };
  int32_t s_strides[1] = { 3 };
  int32_t ref_li[1] = { 5 };

  Tensor input_tensor = new BufferTensor({ 7 }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ 1 }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ 1 }, i32, s_end);
  int32_t begin_mask = 0;
  int32_t end_mask = 0;
  Tensor strides_tensor = new BufferTensor({ 1 }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), 1);
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}


/*
  Random Generated Test Number 03
*/
TEST(StridedIterator, test_strided_it_03) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
  int32_t s_begin[1] = { 0 };
  int32_t s_end[1] = { 9 };
  int32_t s_strides[1] = { 1 };
  int32_t ref_li[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

  Tensor input_tensor = new BufferTensor({ 9 }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ 1 }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ 1 }, i32, s_end);
  int32_t begin_mask = 1;
  int32_t end_mask = 0;
  Tensor strides_tensor = new BufferTensor({ 1 }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), 9);
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}


/*
  Random Generated Test Number 04
*/
TEST(StridedIterator, test_strided_it_04) {
  localCircularArenaAllocator<1024> meta_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);

  float s_input[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  int32_t s_begin[2] = { 0, 1 };
  int32_t s_end[2] = { 1, 5 };
  int32_t s_strides[2] = { 1, 3 };
  int32_t ref_li[2] = { 1, 4 };

  Tensor input_tensor = new BufferTensor({ 2, 5 }, flt, s_input);
  Tensor begin_tensor = new BufferTensor({ 2 }, i32, s_begin);
  Tensor end_tensor = new BufferTensor({ 2 }, i32, s_end);
  int32_t begin_mask = 0;
  int32_t end_mask = 2;
  Tensor strides_tensor = new BufferTensor({ 2 }, i32, s_strides);

  StridedIterator stride_it(input_tensor, begin_tensor, end_tensor,
                            strides_tensor, begin_mask, end_mask);
  EXPECT_EQ(stride_it.num_elems(), 2);
  // check linear index values
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
  EXPECT_EQ(stride_it.next(), -1);  // end of iteration
  // iterate over again
  for (size_t i = 0; i < stride_it.num_elems(); ++i) {
    int32_t li = stride_it.next();
    EXPECT_EQ(li, ref_li[i]);
  }
}
