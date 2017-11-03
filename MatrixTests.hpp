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

    // actual implementation, uses ReferenceGemm()
    // See gen_math_op.py:1619
    // See quantized_matmul_ops.cc:171, 178
    // Sub-functions: QuantizationRangeForMultiplication,
    // QuantizationRangeForMultiplication, FloatForOneQuantizedLevel

    Tensor* out_c = new RamTensor<int>(c->getShape());
    Tensor* out_min = new RamTensor<float>(c_min->getShape());
    Tensor* out_max = new RamTensor<float>(c_max->getShape());
    timer_start();
    QuantizedMatMul<uint8_t, uint8_t, int>(a, b, out_c, a_min, b_min, a_max,
                                           b_max, out_min, out_max);
    timer_stop(); 
    //
    // transpose_a=None, transpose_b=None

    // modify the checks below:

    double result = meanPercentErr<int>(c, out_c) + meanPercentErr<float>(c_min, out_min) +
                    meanPercentErr<float>(c_max, out_max);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) { qMatMul(); }
};
#endif
