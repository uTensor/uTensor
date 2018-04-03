#include "gtest/gtest.h"
#include "tensor.hpp"

TEST(Dummy, One) {
    EXPECT_EQ(0, 0);
}

TEST(Dummy, Two) {
    EXPECT_EQ(0, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    auto out = RUN_ALL_TESTS();

#ifdef _MSC_VER
    system("pause");
#endif

    return out;
}
