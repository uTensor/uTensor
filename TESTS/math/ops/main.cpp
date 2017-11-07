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
#include "MathOps.hpp"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

using namespace utest::v1;

void requantization_rangeTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.int_import("/fs/testData/rqRange/in/qMatMul_0.idx");
    Tensor* a_min =
        t_import.float_import("/fs/testData/rqRange/in/qMatMul_1.idx");
    Tensor* a_max =
        t_import.float_import("/fs/testData/rqRange/in/qMatMul_2.idx");

    // reference outputs
    Tensor* ref_min =
        t_import.float_import("/fs/testData/rqRange/out/rqRange_0.idx");
    Tensor* ref_max =
        t_import.float_import("/fs/testData/rqRange/out/rqRange_1.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor* out_min = new RamTensor<float>(ref_min->getShape());
    Tensor* out_max = new RamTensor<float>(ref_max->getShape());
    //timer_start();
    Requantization_Range<int, float>(a, a_min, a_max, out_min, out_max);
    //timer_stop();

    double result =
        meanPercentErr<float>(ref_min, out_min) + meanPercentErr<float>(ref_max, out_max);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void requantizeTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a = t_import.int_import("/fs/testData/rQ/in/qMatMul_0.idx");
    Tensor* a_min =
        t_import.float_import("/fs/testData/rQ/in/qMatMul_1.idx");
    Tensor* a_max =
        t_import.float_import("/fs/testData/rQ/in/qMatMul_2.idx");
    Tensor* r_a_min =
        t_import.float_import("/fs/testData/rQ/in/rqRange_0.idx");
    Tensor* r_a_max =
        t_import.float_import("/fs/testData/rQ/in/rqRange_1.idx");
    // tf.quint8

    // reference outputs
    Tensor* ref_a_q =
        t_import.ubyte_import("/fs/testData/rQ/out/rQ_0.idx");
    Tensor* ref_a_min =
        t_import.float_import("/fs/testData/rQ/out/rQ_1.idx");
    Tensor* ref_a_max =
        t_import.float_import("/fs/testData/rQ/out/rQ_2.idx");

    // modify the checks below:
    Tensor* a_q = new RamTensor<unsigned char>(ref_a_q->getShape());
    Tensor* a_min_q = new RamTensor<float>(ref_a_min->getShape());
    Tensor* a_max_q = new RamTensor<float>(ref_a_max->getShape());

    // Implementation goes here
    //timer_start();
    Requantize<int, float, unsigned char>(a, a_min, a_max, r_a_min, r_a_max,
            a_q, a_min_q, a_max_q);
    //timer_stop();

    double result = meanPercentErr<unsigned char>(ref_a_q, a_q) +
        meanPercentErr<float>(ref_a_min, a_min_q) +
        meanPercentErr<float>(ref_a_max, a_max_q);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void requantizeTest2(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor*  a = t_import.int_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_0.idx");
    Tensor* a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_1.idx");
    Tensor* a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_2.idx");
    Tensor* r_a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_0.idx");
    Tensor* r_a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_1.idx");
    // tf.quint8

    // reference outputs
    Tensor* ref_a_q =
        t_import.ubyte_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_0.idx");
    Tensor* ref_a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_1.idx");
    Tensor* ref_a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_2.idx");


    // modify the checks below:
    Tensor* a_q = new RamTensor<unsigned char>(ref_a_q->getShape());
    Tensor* a_min_q = new RamTensor<float>(ref_a_min->getShape());
    Tensor* a_max_q = new RamTensor<float>(ref_a_max->getShape());

    // Implementation goes here
    //timer_start();
    Requantize<int, float, unsigned char>(a, a_min, a_max, r_a_min, r_a_max,
            a_q, a_min_q, a_max_q);
    //timer_stop();

    double result;
    if((result = meanPercentErr<unsigned char>(ref_a_q, a_q)) != 0) {
        printf("Requantize a_q failed (%.6f)\r\n", result);
        unsigned char* ref_ptr = ref_a_q->write<unsigned char>(0, 0);
        unsigned char* test_ptr = a_q->write<unsigned char>(0, 0);
        for(uint32_t i = 0; i < ref_a_q->getSize(); i++) {
            if(ref_ptr[i] != test_ptr[i]) {
                printf("%lu: %d != %d\r\n", i, ref_ptr[i], test_ptr[i]);
            } else {
                printf("%lu: %d == %d\r\n", i, ref_ptr[i], test_ptr[i]);
            }
        }
    }


    if((result = meanPercentErr<float>(ref_a_min, a_min_q)) != 0) printf("Requantize a_min_q failed (%.6f)\r\n", result);

    if((result = meanPercentErr<float>(ref_a_max, a_max_q)) != 0) printf("Requantize a_max_q failed (%.6f)\r\n", result);

    result = meanPercentErr<unsigned char>(ref_a_q, a_q) +
        meanPercentErr<float>(ref_a_min, a_min_q) +
        meanPercentErr<float>(ref_a_max, a_max_q);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void argmaxTest(void) {  // NT: WIP   do not use t_import int 64 here
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* ref_a = t_import.float_import("/fs/testData/ArgMax/in/ArgMax-input_0.idx");
    Tensor* ref_dim = t_import.int_import("/fs/testData/ArgMax/in/ArgMax-dimension_0.idx");

    // reference outputs
    /// NT: FIXME: argmax outputs int64 tensor which isn't supported by
    /// int_import.
    Tensor* ref_out = t_import.float_import("/fs/testData/ArgMax/out/ArgMax_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor* out = new RamTensor<int>(ref_out->getShape());

    //timer_start();
    ArgMax<float, int>(ref_a, ref_dim, &out);
    //timer_stop();

    Tensor* out_float = TensorCast<int, float>(out);

    double result = meanPercentErr<float>(ref_out, out_float);

    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void argmaxTest2(void) {  // NT: WIP   do not use t_import int 64 here
    
    Tensor* test_input = TensorConstant<float>({10, 5}, 0.0f);
    *(test_input->write<float>(25, 0)) = 1.0f;
    *(test_input->write<float>(26, 0)) = 1.0f;
    *(test_input->write<float>(7, 0)) = 1.0f;
    *(test_input->write<float>(48, 0)) = 1.0f;
    *(test_input->write<float>(14, 0)) = 1.0f;

    Tensor* test_dim = new RamTensor<int>({1});
    *(test_dim->write<int>(0, 0)) = 0;

    Tensor* test_out_ref = new RamTensor<float>({5});
    *(test_out_ref->write<float>(0, 0)) = 5.0f;
    *(test_out_ref->write<float>(1, 0)) = 5.0f;
    *(test_out_ref->write<float>(2, 0)) = 1.0f;
    *(test_out_ref->write<float>(3, 0)) = 9.0f;
    *(test_out_ref->write<float>(4, 0)) = 2.0f;

    Tensor* test_out = new RamTensor<float>(test_out_ref->getShape());
    //timer_start();
    ArgMax<float, float>(test_input, test_dim, &test_out);
    //timer_stop();

    double result = meanPercentErr<float>(test_out_ref, test_out);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void addTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.float_import("/fs/testData/ref_add/in/Const_5_0.idx");
    Tensor* b =
        t_import.float_import("/fs/testData/ref_add/in/Const_6_0.idx");

    // reference outputs
    Tensor* ref_out =
        t_import.float_import("/fs/testData/ref_add/out/ref_add_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor* out = new RamTensor<float>(ref_out->getShape());
    //timer_start();
    Add<float, float>(a, b, &out);
    //timer_stop();

    double result = meanPercentErr<float>(ref_out, out);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void minTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.float_import("/fs/testData/ref_min/in/Const_2_0.idx");
    Tensor* dim =
        t_import.int_import("/fs/testData/ref_min/in/Const_3_0.idx");

    // reference outputs
    Tensor* ref_out =
        t_import.float_import("/fs/testData/ref_min/out/ref_min_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor* out = new RamTensor<float>(ref_out->getShape());
    //timer_start();
    Min<float, int, float>(a, dim, out);
    //timer_stop();

    double result = meanPercentErr<float>(ref_out, out);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}

void maxTest(void) {
    
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.float_import("/fs/testData/ref_max/in/Const_2_0.idx");
    Tensor* dim =
        t_import.int_import("/fs/testData/ref_max/in/Const_4_0.idx");

    // reference outputs
    Tensor* ref_out =
        t_import.float_import("/fs/testData/ref_max/out/ref_max_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor *out = new RamTensor<float>(ref_out->getShape());
    //timer_start();
    Max<float, int, float>(a, dim, out);
    //timer_stop();

    double result = meanPercentErr<float>(ref_out, out);
    // TEST_ASSERT(result < 0.0001);
    TEST_ASSERT(result == 0);
}
Case cases[] = {
    Case("Requatize range", requantization_rangeTest),
    Case("Requatize 1", requantizeTest),
    Case("Requatize 2", requantizeTest2),
    Case("Argmax 1", argmaxTest),
    Case("Argmax 2", argmaxTest2),
    Case("Add", addTest),
    Case("Min", minTest),
    Case("Max", maxTest),
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
