#include "gtest/gtest.h"
#include "types.hpp"


TEST(Shapes, test_1) {
  TensorShape s1(10);
  EXPECT_EQ(s1.num_dims(), 1);
  EXPECT_EQ(s1[0], 10);
}

TEST(Shapes, test_2d) {
  TensorShape s1(10, 10);
  EXPECT_EQ(s1.num_dims(), 2);
  EXPECT_EQ(s1[0], 10);
  EXPECT_EQ(s1[1], 10);
  EXPECT_EQ(s1.linear_index(1,1,0,0), 11);
}

TEST(Shapes, test_3d) {
  TensorShape s1(10, 10, 10);
  EXPECT_EQ(s1.num_dims(), 3);
  EXPECT_EQ(s1[0], 10);
  EXPECT_EQ(s1[1], 10);
  EXPECT_EQ(s1[2], 10);
  EXPECT_EQ(s1.linear_index(1,1,0,0), 110);
  EXPECT_EQ(s1.linear_index(1,1,1,0), 111);
}
