#include "test_dummy_sanity.h"
#include "test_helper.h"

void test_sanity_check_true(){
    TEST_ASSERT(true);
}

void test_sanity_check_equal(){
    TEST_ASSERT(1 == 1);
}

// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(sanity, check_true, "Sanity check number 1")
UTENSOR_TEST(sanity, check_equal, "Sanity check number 2")

// Third, run like hell
UTENSOR_TEST_RUN()
