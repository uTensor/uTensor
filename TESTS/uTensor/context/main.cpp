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

#include "tensor.hpp"
#include "context.hpp"
#include "MatrixOps.hpp"
#include "deep_mnist_mlp.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");
TensorIdxImporter t_import;

using namespace utest::v1;
void matMulTest(void) {
    
    //inputs
    Tensor* a =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx");
    Tensor* a_min =
        t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx");
    Tensor* a_max =
        t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx");
    Tensor* b =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx");
    Tensor* b_min =
        t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx");
    Tensor* b_max =
        t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");

    // reference outputs
    Tensor* c =
        t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx");
    Tensor* c_min =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx");
    Tensor* c_max =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx");


    Tensor* out_c = new RamTensor<int>(c->getShape());
    Tensor* out_min = new RamTensor<float>(c_min->getShape());
    Tensor* out_max = new RamTensor<float>(c_max->getShape());

    TList inputs = {a, a_min, a_max, b, b_min, b_max};
    TList outputs = {out_c, out_min, out_max};
    Operator* matMal = new QntMatMulOp();

    Context ctx;
    //timer_start();
    ctx.push(matMal, inputs, outputs);
    ctx.eval();
    //timer_stop();

    double result = meanPercentErr<int>(c, out_c) + meanPercentErr<float>(c_min, out_min) +
        meanPercentErr<float>(c_max, out_max);

    TEST_ASSERT(result == 0);
}

Case cases[] = {
    Case("QntMatMal Operator", matMulTest),
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
