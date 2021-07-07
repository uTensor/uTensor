#include "gtest/gtest.h"
#include "uTensor.h"
#include "uTensor/util/broadcast_utils.hpp"

using namespace uTensor;
SimpleErrorHandler mErrHandler(10);

// test broadcastable
TEST(Util, test_broadcastable_01) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(3);
  TensorShape shape2(3, 2);
  EXPECT_EQ(Broadcaster::broadcastable(shape1, shape2), false);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_broadcastable_02) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(1);
  TensorShape shape2(3, 2);
  EXPECT_EQ(Broadcaster::broadcastable(shape1, shape2), true);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_broadcastable_03) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(3, 1, 2);
  TensorShape shape2(1, 2, 3);
  EXPECT_EQ(Broadcaster::broadcastable(shape1, shape2), false);
  EXPECT_EQ(got_error, false);
}

// test promote shpae
TEST(Util, test_promote_shape_01) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(1);
  TensorShape shape2(3, 2);
  Broadcaster broadcaster(shape1, shape2);
  TensorShape promo_shape = broadcaster.promoted_shape();
  TensorShape target_shape(3, 2);
  EXPECT_EQ(promo_shape, target_shape);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_promote_shape_02) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(1);
  TensorShape shape2(1, 2);
  Broadcaster broadcaster(shape1, shape2);
  TensorShape promo_shape = broadcaster.promoted_shape();
  TensorShape target_shape(1, 2);
  EXPECT_EQ(promo_shape, target_shape);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_promote_shape_03) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(3, 1, 2);
  TensorShape shape2(1, 2, 1);
  Broadcaster broadcaster(shape1, shape2);
  TensorShape promo_shape = broadcaster.promoted_shape();
  TensorShape target_shape(3, 2, 2);
  EXPECT_EQ(promo_shape, target_shape);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_linear_idx_01) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(1);
  TensorShape shape2(3, 2);
  Broadcaster brd(shape1, shape2);
  int32_t ans1[6] = {0, 0, 0, 0, 0, 0};
  int32_t ans2[6] = {0, 1, 2, 3, 4, 5};
  int32_t li1 = 0, li2 = 0;
  for (size_t i = 0; i < 6; ++i) {
    brd.next(li1, li2);
    EXPECT_EQ(li1, ans1[i]);
    EXPECT_EQ(li2, ans2[i]);
  }
  brd.next(li1, li2);
  EXPECT_EQ(li1, -1);
  EXPECT_EQ(li2, -1);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_linear_idx_02) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(3, 1);
  TensorShape shape2(3, 2);
  Broadcaster brd(shape1, shape2);
  int32_t ans1[6] = {0, 0, 1, 1, 2, 2};
  int32_t ans2[6] = {0, 1, 2, 3, 4, 5};
  int32_t li1 = 0, li2 = 0;
  for (size_t i = 0; i < 6; ++i) {
    brd.next(li1, li2);
    EXPECT_EQ(li1, ans1[i]);
    EXPECT_EQ(li2, ans2[i]);
  }
  brd.next(li1, li2);
  EXPECT_EQ(li1, -1);
  EXPECT_EQ(li2, -1);
  EXPECT_EQ(got_error, false);
}

TEST(Util, test_linear_idx_03) {
  bool got_error = false;
  mErrHandler.set_onError([&got_error](Error* err) { got_error = true; });
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  TensorShape shape1(2, 1, 2);
  TensorShape shape2(1, 2, 1);
  Broadcaster brd(shape1, shape2);
  int32_t ans1[8] = {0, 1, 0, 1, 2, 3, 2, 3};
  int32_t ans2[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  int32_t li1 = 0, li2 = 0;
  for (size_t i = 0; i < 8; ++i) {
    brd.next(li1, li2);
    EXPECT_EQ(li1, ans1[i]);
    EXPECT_EQ(li2, ans2[i]);
  }
  brd.next(li1, li2);
  EXPECT_EQ(li1, -1);
  EXPECT_EQ(li2, -1);
  EXPECT_EQ(got_error, false);
}