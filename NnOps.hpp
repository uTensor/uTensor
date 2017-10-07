#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include <test.hpp>

class NnOpsTest : public Test {
public:
    void reluTest(void) {
        testStart("relu");
        TensorIdxImporter t_import;

        //reference inputs
        Tensor<float> a = t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx");
        Tensor<float> min = t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx");
        Tensor<float> max = t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx");

        //reference outputs
        Tensor<float> ref_out = t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx");
        Tensor<float> ref_min = t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx");
        Tensor<float> ref_max = t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx");

        //Implementation goes here

        //modify the checks below:
        Tensor<float> out(ref_out.getShape());
        Tensor<float> out_min(ref_out.getShape());
        Tensor<float> out_max(ref_out.getShape());
    

        double result = meanPercentErr(ref_out, out) + meanPercentErr(ref_min, out_min) + meanPercentErr(ref_max, out_max);
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void runAll(void) {
        reluTest();
    }
};


#endif //UTENSOR_NN_OPS