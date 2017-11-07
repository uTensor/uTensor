#include <cstdio>
#include "greentea-client/test_env.h"
#include "mbed.h"
#include "utest.h"
#include "unity.h"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "uTensor_util.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

using namespace utest::v1;

void ntoh32Test(void) {

    uint32_t input = 63;
    //timer_start();
    uint32_t result = ntoh32(input);
    //timer_stop();
    TEST_ASSERT(result == 1056964608);
}

void ucharTest(void) {

    TensorIdxImporter t_import;
    //timer_start();
    Tensor* t =
        t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx");
    //timer_stop();
    double result = sum<unsigned char>(t);
    TEST_ASSERT(result == 4518);
    delete t;
}

void shortTest(void) {

    TensorIdxImporter t_import;
    //timer_start();
    Tensor* t =
        t_import.short_import("/fs/testData/idxImport/int16_4d_power2.idx");
    //timer_stop();
    double result = sum<short>(t);
    TEST_ASSERT(result == 270250);
    delete t;
}

void intTest(void) {

    TensorIdxImporter t_import;
    //timer_start();
    Tensor* t =
        t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    //timer_stop();
    double result = sum<int>(t);
    TEST_ASSERT(result == 5748992600);
    delete t;
}

void floatTest(void) {

    TensorIdxImporter t_import;
    //timer_start();
    Tensor* t =
        t_import.float_import("/fs/testData/idxImport/float_4d_power2.idx");
    //timer_stop();

    double result = sum<float>(t);

    DEBUG("***floating point test yielded: %.8e\r\n", (float)result);
    TEST_ASSERT((float)result == -1.0f);
    delete t;
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
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    printf("Deep MLP on Mbed (Trained with Tensorflow)\r\n\r\n");
    printf("running deep-mlp...\r\n");

    int prediction = runMLP("/fs/testData/deep_mlp/import-Placeholder_0.idx");
    printf("prediction: %d\r\n", prediction);
    return Harness::run(specification);
}
