#ifndef UTENSOR_NN_TESTS
#define UTENSOR_NN_TESTS

#include "NnOps.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"

class NnOpsTest : public Test {
 public:
  void reluTest(void) {
    testStart("quantized_relu");
    TensorIdxImporter t_import;

    // reference inputs
    Tensor* a =
        t_import.ubyte_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx");
    Tensor* min =
        t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx");
    Tensor* max =
        t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx");

    // reference outputs
    Tensor* ref_out =
        t_import.ubyte_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx");
    Tensor* ref_min =
        t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx");
    Tensor* ref_max =
        t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx");

    // modify the checks below:
    Tensor* out = new RamTensor<unsigned char>(ref_out->getShape());
    Tensor* out_min = new RamTensor<float>(ref_min->getShape());
    Tensor* out_max = new RamTensor<float>(ref_max->getShape());

    timer_start();
    Relu<unsigned char, float, unsigned char>(a, min, max, out, out_min,
                                              out_max);
    timer_stop();

    double result = meanPercentErr<unsigned char>(ref_out, out) +
                    meanPercentErr<float>(ref_min, out_min) +
                    meanPercentErr<float>(ref_max, out_max);
    // passed(result < 0.0001);
    passed(result == 0);
    delete a;
    delete min;
    delete max;
    delete ref_out;
    delete ref_min;
    delete ref_max;
    delete out;
    delete out_min;
    delete out_max;
  }

  void runAll(void) { reluTest(); }
};

#endif  // UTENSOR_NN_TESTS
