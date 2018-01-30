#ifndef UTENSOR_NN_TESTS
#define UTENSOR_NN_TESTS

#include "NnOps.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"

class NnOpsTest : public Test {
  Context ctx;
  TensorIdxImporter t_import;

 public:
  void reluTest(void) {
    testStart("quantized_relu");
    // reference inputs
    S_TENSOR a =
        ctx.add(t_import.ubyte_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx"), "a");
    S_TENSOR min =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx"), "min");
    S_TENSOR max =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx"), "max");

    // reference outputs
    S_TENSOR ref_out =
        ctx.add(t_import.ubyte_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx"), "ref_out");
    S_TENSOR ref_min =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx"), "ref_min");
    S_TENSOR ref_max =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx"), "ref_max");

    // modify the checks below:
    S_TENSOR out = ctx.add(new RamTensor<unsigned char>(ref_out->getShape()), "out");
    S_TENSOR out_min = ctx.add(new RamTensor<float>(ref_min->getShape()), "out_min");
    S_TENSOR out_max = ctx.add(new RamTensor<float>(ref_max->getShape()), "out_max");


    timer_start();
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), {"a", "min", "max"}, {"out", "out_min", "out_max"});
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<unsigned char>(ref_out.get(), out.get()) +
                    meanPercentErr<float>(ref_min.get(), out_min.get()) +
                    meanPercentErr<float>(ref_max.get(), out_max.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) { reluTest(); }
};

#endif  // UTENSOR_NN_TESTS
