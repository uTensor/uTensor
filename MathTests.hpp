#ifndef UTENSOR_MATH_TESTS
#define UTENSOR_MATH_TESTS

#include "MathOps.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"

class MathOpsTest : public Test {
 public:
  void requantization_rangeTest(void) {
    testStart("requantization_range");
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
    timer_start();
    Requantization_Range<int, float>(a, a_min, a_max, out_min, out_max);
    timer_stop();

    double result =
        meanPercentErr(ref_min, out_min) + meanPercentErr(ref_max, out_max);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void requantizeTest(void) {
    testStart("requantize");
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
    timer_start();
    Requantize<int, float, unsigned char>(a, a_min, a_max, r_a_min, r_a_max,
                                          a_q, a_min_q, a_max_q);
    timer_stop();

    double result = meanPercentErr(ref_a_q, a_q) +
                    meanPercentErr(ref_a_min, a_min_q) +
                    meanPercentErr(ref_a_max, a_max_q);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void argmaxTest(void) {  // NT: WIP   do not use t_import int 64 here
    testStart("argmax");
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
    
    timer_start();
    ArgMax(ref_a, ref_dim, out);
    timer_stop();
    
    Tensor<float> out_float = TensorCast<int, float>(out);

    double result = meanPercentErr(ref_out, out_float);

    // passed(result < 0.0001);
    passed(result == 0);
  }

  void argmaxTest2(void) {  // NT: WIP   do not use t_import int 64 here
    testStart("argmax2");
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
    timer_start();
    ArgMax(test_input, test_dim, test_out);
    timer_stop();

    double result = meanPercentErr(test_out_ref, test_out);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void addTest(void) {
    testStart("add");
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
    timer_start();
    Add<float, float>(a, b, out);
    timer_stop();

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void minTest(void) {
    testStart("min");
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
    timer_start();
    Min<float, int, float>(a, dim, out);
    timer_stop();

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void maxTest(void) {
    testStart("max");
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
    timer_start();
    Max<float, int, float>(a, dim, out);
    timer_stop();

    double result = meanPercentErr(ref_out, out);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) {
    argmaxTest();
    argmaxTest2();
    requantization_rangeTest();
    requantizeTest();
    addTest();
    minTest();
    maxTest();
  }
};

#endif  // UTENSOR_MATH_TESTS
