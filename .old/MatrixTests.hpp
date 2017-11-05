#ifndef UTENSOR_MATRIX_TEST 
#define UTENSOR_MATRIX_TEST 

#include "test.hpp"
#include "MatrixOps.hpp"
#include "tensorIdxImporter.hpp"

class matrixOpsTest : public Test {
 public:
  void qMatMul(void) {
    testStart("Quantized Matrix Mul");
    TensorIdxImporter t_import;

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
    timer_start();
    QuantizedMatMul<uint8_t, uint8_t, int>(a, b, out_c, a_min, b_min, a_max,
                                           b_max, out_min, out_max);
    timer_stop(); 
    //
    // transpose_a=None, transpose_b=None

    // modify the checks below:

    double result = meanPercentErr(c, out_c) + meanPercentErr(c_min, out_min) +
                    meanPercentErr(c_max, out_max);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) { qMatMul(); }
};
#endif
