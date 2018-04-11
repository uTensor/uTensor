#include "test_helper.h"

// Default to using GTest like asserts and expects as these give more info that unity
// We will forward these commands to unity in test_helper.h
void test_sanity_checkTrue(){
    EXPECT_EQ(true, true);
}

void test_sanity_checkEqual(){
    EXPECT_EQ(1 == 1, true);
}

// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(sanity, checkTrue, "Sanity check number 1")
UTENSOR_TEST(sanity, checkEqual, "Sanity check number 2")


// Third, run like hell
UTENSOR_TEST_RUN()
