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
#include "ArrayOps.hpp"
#include "deep_mnist_mlp.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

using namespace utest::v1;

void quantize_v2Test(void) {
    
    TensorIdxImporter t_import;

    //reference inputs  /Users/neitan01/Documents/mbed/uTensor.git/TESTS/scripts/PRE-GEN/qA
    Tensor* b = t_import.float_import ("/fs/testData/qB/in/Cast_1_0.idx");
    Tensor* b_min = t_import.float_import("/fs/testData/qB/in/Min_1_0.idx");
    Tensor* b_max = t_import.float_import("/fs/testData/qB/in/Max_1_0.idx");

    //reference outputs
    Tensor* b_q_ref = t_import.ubyte_import("/fs/testData/qB/out/qB_0.idx");
    Tensor* b_min_q_ref = t_import.float_import("/fs/testData/qB/out/qB_1.idx");
    Tensor* b_max_q_ref = t_import.float_import("/fs/testData/qB/out/qb_2.idx");

    Tensor* b_q = new RamTensor<unsigned char>(b_q_ref->getShape());
    Tensor* b_min_q = new RamTensor<float>(b_min_q_ref->getShape());
    Tensor* b_max_q = new RamTensor<float>(b_max_q_ref->getShape());

    //Implementation goes here
    //timer_start();
    QuantizeV2<unsigned char>(b, b_min, b_max, b_q, b_min_q, b_max_q);
    //timer_stop();

    // printf("refMin is : %f \r\n", *(b_min_q_ref.getPointer({0})));
    // printf("outMin is : %f \r\n", *(b_min_q.getPointer({0})));
    // printf("diff : output(%f), outMin(%f), outMax(%f)\r\n", 
    //  meanPercentErr(b_q_ref, b_q), meanPercentErr(b_min_q_ref, b_min_q), meanPercentErr(b_max_q_ref, b_max_q));

    double result = meanPercentErr<unsigned char>(b_q_ref, b_q) + meanPercentErr<float>(b_min_q_ref, b_min_q) + meanPercentErr<float>(b_max_q_ref, b_max_q);
    //TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void dequantizeTest(void) {
    
    TensorIdxImporter t_import;

    //reference inputs
    Tensor* a = t_import.ubyte_import("/fs/testData/deQ/in/rQ_0.idx");
    Tensor* a_min = t_import.float_import("/fs/testData/deQ/in/rQ_1.idx");
    Tensor* a_max = t_import.float_import("/fs/testData/deQ/in/rQ_2.idx");

    //reference outputs
    Tensor* out_ref = t_import.float_import("/fs/testData/deQ/out/deQ_0.idx");

    //modify the checks below:
    Tensor* out = new RamTensor<float>(out_ref->getShape());

    //timer_start();
    dequantize<unsigned char>(a, a_min, a_max, &out);
    //timer_stop();

    double result = meanPercentErr<float>(out_ref, out);
    //TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void reshapeTest(void) {
    
    TensorIdxImporter t_import;

    //reference inputs
    Tensor* ref_a = t_import.float_import("/fs/testData/ref_reshape/in/Const_0.idx");
    Tensor* ref_dim = t_import.int_import("/fs/testData/ref_reshape/in/Const_1_0.idx");

    //reference outputs
    Tensor* out_ref = t_import.float_import("/fs/testData/ref_reshape/out/ref_reshape_0.idx");

    //modify the checks below:
    Tensor* out = new RamTensor<float>(out_ref->getShape());

    //timer_start();
    reshape<float>(ref_a, ref_dim, &out);
    //timer_stop();

    double result = meanPercentErr<float>(out_ref, out);
    //TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}
Case cases[] = {
    Case("Quantize", quantize_v2Test),
    Case("Dequantize", dequantizeTest),
    Case("Reshape", reshapeTest),
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
