#include "greentea-client/test_env.h"
#include "mbed.h"
#include "utest.h"
#include "unity.h"
#include "MatrixOps.hpp"
#include "tensorIdxImporter.hpp"
#include "uTensorTest.hpp"
#include "MathOps.hpp"
#include "uTensor.hpp"

using namespace utest::v1;
uTensor utensor;

void matrix_ops_test(){
    TensorIdxImporter t_import;
    printf("Here\n\r");

    // reference inputs
    Tensor<unsigned char> a =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx");
    Tensor<float> a_min =
        t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx");
    Tensor<float> a_max =
        t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx");
    Tensor<unsigned char> b =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx");
    Tensor<float> b_min =
        t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx");
    Tensor<float> b_max =
        t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");

    // reference outputs
    Tensor<int> c =
        t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx");
    Tensor<float> c_min =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx");
    Tensor<float> c_max =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx");

    // actual implementation, uses ReferenceGemm()
    // See gen_math_op.py:1619
    // See quantized_matmul_ops.cc:171, 178
    // Sub-functions: QuantizationRangeForMultiplication,
    // QuantizationRangeForMultiplication, FloatForOneQuantizedLevel

    Tensor<int> out_c(c.getShape());
    Tensor<float> out_min(c_min.getShape());
    Tensor<float> out_max(c_max.getShape());
    QuantizedMatMul<uint8_t, uint8_t, int>(a, b, out_c, a_min, b_min, a_max,
            b_max, out_min, out_max);
    //
    // transpose_a=None, transpose_b=None

    // modify the checks below:

    double result = meanPercentErr(c, out_c) + meanPercentErr(c_min, out_min) +
        meanPercentErr(c_max, out_max);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void requantization_range_test(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<int> a =
        t_import.int_import("/fs/testData/rqRange/in/qMatMul_0.idx");
    Tensor<float> a_min =
        t_import.float_import("/fs/testData/rqRange/in/qMatMul_1.idx");
    Tensor<float> a_max =
        t_import.float_import("/fs/testData/rqRange/in/qMatMul_2.idx");

    // reference outputs
    Tensor<float> ref_min =
        t_import.float_import("/fs/testData/rqRange/out/rqRange_0.idx");
    Tensor<float> ref_max =
        t_import.float_import("/fs/testData/rqRange/out/rqRange_1.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor<float> out_min(ref_min.getShape());
    Tensor<float> out_max(ref_max.getShape());
    Requantization_Range<int, float>(a, a_min, a_max, out_min, out_max);

    double result =
        meanPercentErr(ref_min, out_min) + meanPercentErr(ref_max, out_max);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void requantization_test(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<int> a = t_import.int_import("/fs/testData/rQ/in/qMatMul_0.idx");
    Tensor<float> a_min =
        t_import.float_import("/fs/testData/rQ/in/qMatMul_1.idx");
    Tensor<float> a_max =
        t_import.float_import("/fs/testData/rQ/in/qMatMul_2.idx");
    Tensor<float> r_a_min =
        t_import.float_import("/fs/testData/rQ/in/rqRange_0.idx");
    Tensor<float> r_a_max =
        t_import.float_import("/fs/testData/rQ/in/rqRange_1.idx");
    // tf.quint8

    // reference outputs
    Tensor<unsigned char> ref_a_q =
        t_import.ubyte_import("/fs/testData/rQ/out/rQ_0.idx");
    Tensor<float> ref_a_min =
        t_import.float_import("/fs/testData/rQ/out/rQ_1.idx");
    Tensor<float> ref_a_max =
        t_import.float_import("/fs/testData/rQ/out/rQ_2.idx");

    // modify the checks below:
    Tensor<unsigned char> a_q(ref_a_q.getShape());
    Tensor<float> a_min_q(ref_a_min.getShape());
    Tensor<float> a_max_q(ref_a_max.getShape());

    // Implementation goes here
    Requantize<int, float, unsigned char>(a, a_min, a_max, r_a_min, r_a_max,
            a_q, a_min_q, a_max_q);

    double result = meanPercentErr(ref_a_q, a_q) +
        meanPercentErr(ref_a_min, a_min_q) +
        meanPercentErr(ref_a_max, a_max_q);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void requantization_test_2(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<int> a = t_import.int_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_0.idx");
    Tensor<float> a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_1.idx");
    Tensor<float> a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_quantized_mat_mul_2.idx");
    Tensor<float> r_a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_0.idx");
    Tensor<float> r_a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/in/import-MatMul_eightbit_requant_range_1.idx");
    // tf.quint8

    // reference outputs
    Tensor<unsigned char> ref_a_q =
        t_import.ubyte_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_0.idx");
    Tensor<float> ref_a_min =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_1.idx");
    Tensor<float> ref_a_max =
        t_import.float_import("/fs/testData/import-MatMul_eightbit_requantize/out/import-MatMul_eightbit_requantize_2.idx");


    // modify the checks below:
    Tensor<unsigned char> a_q(ref_a_q.getShape());
    Tensor<float> a_min_q(ref_a_min.getShape());
    Tensor<float> a_max_q(ref_a_max.getShape());

    // Implementation goes here
    Requantize<int, float, unsigned char>(a, a_min, a_max, r_a_min, r_a_max,
            a_q, a_min_q, a_max_q);

    double result;
    if((result = meanPercentErr(ref_a_q, a_q)) != 0) {
        printf("Requantize a_q failed (%.6f)\r\n", result);
        unsigned char* ref_ptr = ref_a_q.getPointer({});
        unsigned char* test_ptr = a_q.getPointer({});
        for(uint32_t i = 0; i < ref_a_q.getSize(); i++) {
            if(ref_ptr[i] != test_ptr[i]) {
                printf("%lu: %d != %d\r\n", i, ref_ptr[i], test_ptr[i]);
            } else {
                printf("%lu: %d == %d\r\n", i, ref_ptr[i], test_ptr[i]);
            }
        }
    }


    if((result = meanPercentErr(ref_a_min, a_min_q)) != 0) printf("Requantize a_min_q failed (%.6f)\r\n", result);

    if((result = meanPercentErr(ref_a_max, a_max_q)) != 0) printf("Requantize a_max_q failed (%.6f)\r\n", result);

    result = meanPercentErr(ref_a_q, a_q) +
        meanPercentErr(ref_a_min, a_min_q) +
        meanPercentErr(ref_a_max, a_max_q);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void argmax_test(void) {  // NT: WIP   do not use t_import int 64 here
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<float> ref_a = t_import.float_import("/fs/testData/ArgMax/in/ArgMax-input_0.idx");
    Tensor<int> ref_dim = t_import.int_import("/fs/testData/ArgMax/in/ArgMax-dimension_0.idx");

    // reference outputs
    /// NT: FIXME: argmax outputs int64 tensor which isn't supported by
    /// int_import.
    Tensor<float> ref_out = t_import.float_import("/fs/testData/ArgMax/out/ArgMax_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor<int> out(ref_out.getShape());

    ArgMax(ref_a, ref_dim, out);

    Tensor<float> out_float = TensorCast<int, float>(out);

    double result = meanPercentErr(ref_out, out_float);

    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void argmax_test_2(void) {  // NT: WIP   do not use t_import int 64 here
    Tensor<float> test_input = TensorConstant<float>({10, 5}, 0.0f);
    *(test_input.getPointer({5,0})) = 1.0f;
    *(test_input.getPointer({5,1})) = 1.0f;
    *(test_input.getPointer({1,2})) = 1.0f;
    *(test_input.getPointer({9,3})) = 1.0f;
    *(test_input.getPointer({2,4})) = 1.0f;

    Tensor<int> test_dim({1});
    *(test_dim.getPointer({0})) = 0;

    Tensor<float> test_out_ref({5});
    *(test_out_ref.getPointer({0})) = 5.0f;
    *(test_out_ref.getPointer({1})) = 5.0f;
    *(test_out_ref.getPointer({2})) = 1.0f;
    *(test_out_ref.getPointer({3})) = 9.0f;
    *(test_out_ref.getPointer({4})) = 2.0f;

    Tensor<float> test_out(test_out_ref.getShape());
    ArgMax(test_input, test_dim, test_out);

    double result = meanPercentErr(test_out_ref, test_out);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void add_test(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<float> a =
        t_import.float_import("/fs/testData/ref_add/in/Const_5_0.idx");
    Tensor<float> b =
        t_import.float_import("/fs/testData/ref_add/in/Const_6_0.idx");

    // reference outputs
    Tensor<float> ref_out =
        t_import.float_import("/fs/testData/ref_add/out/ref_add_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor<float> out(ref_out.getShape());
    Add<float, float>(a, b, out);

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void min_test(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<float> a =
        t_import.float_import("/fs/testData/ref_min/in/Const_2_0.idx");
    Tensor<int> dim =
        t_import.int_import("/fs/testData/ref_min/in/Const_3_0.idx");

    // reference outputs
    Tensor<float> ref_out =
        t_import.float_import("/fs/testData/ref_min/out/ref_min_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor<float> out(ref_out.getShape());
    Min<float, int, float>(a, dim, out);

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

void max_test(void) {
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<float> a =
        t_import.float_import("/fs/testData/ref_max/in/Const_2_0.idx");
    Tensor<int> dim =
        t_import.int_import("/fs/testData/ref_max/in/Const_4_0.idx");

    // reference outputs
    Tensor<float> ref_out =
        t_import.float_import("/fs/testData/ref_max/out/ref_max_0.idx");

    // Implementation goes here

    // modify the checks below:
    Tensor<float> out(ref_out.getShape());
    Max<float, int, float>(a, dim, out);

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    TEST_ASSERT_EQUAL(result, 0);
}

Case cases[] = {
    Case("matrix_ops_test", matrix_ops_test),
    Case("requantization_range_test", requantization_range_test),
    Case("requantization_test", requantization_test),
    Case("requantization_test_2", requantization_test_2),
    Case("argmax_test", argmax_test),
    Case("argmax_test_2", argmax_test_2),
    Case("add_test", add_test),
    Case("min_test", min_test),
    Case("max_test", max_test)
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
