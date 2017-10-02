#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include <test.hpp>

class MathOpsTest : public Test {
public:
    void requantization_rangeTest(void) {
        testStart("requantization_range");
        TensorIdxImporter t_import;

        //reference inputs
        Tensor<int> a = t_import.int_import("/fs/testData/rqRange/in/qMatMul_0.idx");
        Tensor<float> a_min = t_import.float_import("/fs/testData/rqRange/in/qMatMul_1.idx");
        Tensor<float> a_max = t_import.float_import("/fs/testData/rqRange/in/qMatMul_2.idx");

        //reference outputs
        Tensor<float> r_a_min_ref = t_import.float_import("/fs/testData/rqRange/out/rqRange_0.idx");
        Tensor<float> r_a_max_ref = t_import.float_import("/fs/testData/rqRange/out/rqRange_1.idx");

        //Implementation goes here

        //modify the checks below:
        Tensor<float> r_a_min(r_a_min_ref.getShape());
        Tensor<float> r_a_max(r_a_max_ref.getShape());

        double result = meanPercentErr(r_a_min_ref, r_a_min) + meanPercentErr(r_a_max_ref, r_a_max);
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void requantizeTest(void) {
        testStart("requantize");
        TensorIdxImporter t_import;
        
        //reference inputs
        Tensor<int> a = t_import.int_import ("/fs/testData/rQ/in/qMatMul_0.idx");
        Tensor<float> a_min = t_import.float_import("/fs/testData/rQ/in/qMatMul_1.idx");
        Tensor<float> a_max = t_import.float_import("/fs/testData/rQ/in/qMatMul_2.idx");
        Tensor<float> r_a_min = t_import.float_import("/fs/testData/rQ/in/rqRange_0.idx");
        Tensor<float> r_a_max = t_import.float_import("/fs/testData/rQ/in/rqRange_1.idx");
        //tf.quint8

        //reference outputs
        Tensor<unsigned char> a_q_ref = t_import.ubyte_import("/fs/testData/rQ/out/rQ_0.idx");
        Tensor<float> a_min_q_ref = t_import.float_import("/fs/testData/rQ/out/rQ_1.idx");
        Tensor<float> a_max_q_ref = t_import.float_import("/fs/testData/rQ/out/rQ_2.idx");

        //Implementation goes here

        //modify the checks below:
        Tensor<unsigned char> a_q(a_q_ref.getShape());
        Tensor<float> a_min_q(a_min_q_ref.getShape());
        Tensor<float> a_max_q(a_max_q_ref.getShape());

        double result = meanPercentErr(a_q_ref, a_q) + meanPercentErr(a_min_q_ref, a_min_q) + meanPercentErr(a_max_q_ref, a_max_q);
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void runAll(void) {
        requantization_rangeTest();
        requantizeTest();
    }
};


#endif //UTENSOR_MATH_OPS