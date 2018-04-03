#include "test_helper.h"

void test_sanity_checkTrue(){
    TEST_ASSERT(true);
}

void test_sanity_checkEqual(){
    TEST_ASSERT(1 == 1);
}

// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(sanity, checkTrue, "Sanity check number 1")
UTENSOR_TEST(sanity, checkEqual, "Sanity check number 2")


// Third, run like hell
UTENSOR_TEST_RUN()
