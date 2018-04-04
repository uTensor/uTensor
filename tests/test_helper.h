#ifndef __TEST_HELPER_H__
#define __TEST_HELPER_H__
#include "unity.h"

#ifdef COMPILE_GREENTEA
// Handle all the boilerplate code
#include "mbed.h"	
#include "greentea-client/test_env.h"	
#include "unity.h"	
#include "utest.h"	
#include <vector>
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
	
using namespace utest::v1;

// Custom setup handler required for proper Greentea support
utest::v1::status_t greentea_setup(const size_t number_of_cases) {
    //Timeout 20
    GREENTEA_SETUP(20, "default_auto");
    // Call the default reporting function
    return greentea_test_setup_handler(number_of_cases);
}

#define UTENSOR_TEST_CONFIGURE() std::vector<Case> cases(); \
    Serial pc(USBTX, USBRX, 115200); \
+SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS); \
+FATFileSystem fs("fs");

#define UTENSOR_TEST(x, y, message) cases.push_back(Case( message, test_ ## x ## _ ## y ));

#define UTENSOR_TEST_RUN() Specification specification(greentea_setup, cases.data); \
    int main(){ \
        ON_ERR(bd.init(), "SDBlockDevice init "); \
        ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". "); \
        return Harness::run(specification); \
    }


#else /* GTest */

#include "gtest/gtest.h"
#define UTENSOR_TEST_CONFIGURE() /* pass */

#define UTENSOR_TEST(x, y, message) GTEST_TEST(x, y){ test_ ## x ## _ ## y(); }

// Google does this better than I can
#define UTENSOR_TEST_RUN() int main(int argc, char** argv) { \
        ::testing::InitGoogleTest(&argc, argv); \
        auto out = RUN_ALL_TESTS(); \
        return out; \
    }

#endif

#endif
