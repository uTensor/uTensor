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

void reluTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.ubyte_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx");
    Tensor* min =
        t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx");
    Tensor* max =
        t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx");

    // reference outputs
    Tensor* ref_out =
        t_import.ubyte_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx");
    Tensor* ref_min =
        t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx");
    Tensor* ref_max =
        t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx");

    // modify the checks below:
    Tensor* out = new RamTensor<unsigned char>(ref_out->getShape());
    Tensor* out_min = new RamTensor<float>(ref_min->getShape());
    Tensor* out_max = new RamTensor<float>(ref_max->getShape());

    //timer_start();
    Relu<unsigned char, float, unsigned char>(a, min, max, out, out_min,
            out_max);
    //timer_stop();

    double result = meanPercentErr<unsigned char>(ref_out, out) +
        meanPercentErr<float>(ref_min, out_min) +
        meanPercentErr<float>(ref_max, out_max);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
    delete a;
    delete min;
    delete max;
    delete ref_out;
    delete ref_min;
    delete ref_max;
    delete out;
    delete out_min;
    delete out_max;
}
Case cases[] = {
    Case("ReLu", reluTest),
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
