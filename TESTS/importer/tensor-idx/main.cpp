#include "greentea-client/test_env.h"
#include "mbed.h"
#include "utest.h"
#include "unity.h"
#include "tensorIdxImporter.hpp"
#include "uTensorTest.hpp"
#include "uTensor.hpp"

using namespace utest::v1;

uTensor utensor;

void ntoh32Test(void) {
    //testStart("ntoh32 test");
    uint32_t input = 63;
    //timer_start();
    uint32_t result = ntoh32(input);
    //timer_stop();
    TEST_ASSERT(result == 1056964608);
}

void ucharTest(void) {
    //testStart("uchar import test");
    TensorIdxImporter t_import;
    //timer_start();
    Tensor<unsigned char> t =
        t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx");
    //timer_stop();
    double result = sum(t);
    TEST_ASSERT(result == 4518);
}

void shortTest(void) {
    //testStart("short import test");
    TensorIdxImporter t_import;
    //timer_start();
    Tensor<short> t =
        t_import.short_import("/fs/testData/idxImport/int16_4d_power2.idx");
    //timer_stop();
    double result = sum(t);
    TEST_ASSERT(result == 270250);
}

void intTest(void) {
    //testStart("int import test");
    TensorIdxImporter t_import;
    //timer_start();
    Tensor<int> t =
        t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    //timer_stop();
    double result = sum(t);
    TEST_ASSERT(result == 5748992600);
}

void floatTest(void) {
    //testStart("float import test");
    TensorIdxImporter t_import;
    //timer_start();
    Tensor<float> t =
        t_import.float_import("/fs/testData/idxImport/float_4d_power2.idx");
    //timer_stop();

    double result = sum(t);

    DEBUG("***floating point test yielded: %.8e\r\n", (float)result);
    // Need to do proper float comparison
    TEST_ASSERT((float)result == -1.0f);
}

Case cases[] = {
    Case("ntoh32Test", ntoh32Test),
    Case("ucharTest", ucharTest),
    Case("shortTest", shortTest),
    Case("intTest", intTest),
    Case("floatTest", floatTest),
};

// Custom setup handler required for proper Greentea support
utest::v1::status_t greentea_setup(const size_t number_of_cases) {
    //Timeout 20
    GREENTEA_SETUP(20, "default_auto");
    // Call the default reporting function
    return greentea_test_setup_handler(number_of_cases);
}

Specification specification(greentea_setup, cases);

int main(){
    return Harness::run(specification);
}
