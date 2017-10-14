#ifndef UTENSOR_NN_TESTS
#define UTENSOR_NN_TESTS

#include "test.hpp"
#include "NnOps.hpp"
#include "tensorIdxImporter.hpp"

class NnOpsTest : public Test {
public:
    void reluTest(void) {
        testStart("quantized_relu");
        TensorIdxImporter t_import;

        //reference inputs
        Tensor<unsigned char> a = t_import.ubyte_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx");
        Tensor<float> min = t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx");
        Tensor<float> max = t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx");

        //reference outputs
        Tensor<unsigned char> ref_out = t_import.ubyte_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx");
        Tensor<float> ref_min = t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx");
        Tensor<float> ref_max = t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx");

        //modify the checks below:
        Tensor<unsigned char> out(ref_out.getShape());
        Tensor<float> out_min(ref_out.getShape());
        Tensor<float> out_max(ref_out.getShape());

        timer_start();
        //Implementation goes here
        timer_stop();


        double result = meanPercentErr(ref_out, out) + meanPercentErr(ref_min, out_min) + meanPercentErr(ref_max, out_max);
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void runAll(void) {
        reluTest();
    }
};


#endif //UTENSOR_NN_TESTS