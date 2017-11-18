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
    TENSOR a =
        ctx.add(t_import.ubyte_import("/fs/testData/ref_qRelu/in/QuantizeV2_0.idx"));
    TENSOR min =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_1.idx"));
    TENSOR max =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/in/QuantizeV2_2.idx"));

    // reference outputs
    TENSOR ref_out =
        ctx.add(t_import.ubyte_import("/fs/testData/ref_qRelu/out/ref_qRelu_0.idx"));
    TENSOR ref_min =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_1.idx"));
    TENSOR ref_max =
        ctx.add(t_import.float_import("/fs/testData/ref_qRelu/out/ref_qRelu_2.idx"));

    // modify the checks below:
    TENSOR out = ctx.add(new RamTensor<unsigned char>(ref_out.lock()->getShape()));
    TENSOR out_min = ctx.add(new RamTensor<float>(ref_min.lock()->getShape()));
    TENSOR out_max = ctx.add(new RamTensor<float>(ref_max.lock()->getShape()));

    //lock on to required output tensors
    S_TENSOR ref_out_s = ref_out.lock();
    S_TENSOR ref_min_s = ref_min.lock();
    S_TENSOR ref_max_s = ref_max.lock();
    S_TENSOR out_s = out.lock();
    S_TENSOR out_min_s = out_min.lock();
    S_TENSOR out_max_s = out_max.lock();

    TList inputs = {a, min, max};
    TList outputs = {out, out_min, out_max};

    timer_start();
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<unsigned char>(ref_out_s.get(), out_s.get()) +
                    meanPercentErr<float>(ref_min_s.get(), out_min_s.get()) +
                    meanPercentErr<float>(ref_max_s.get(), out_max_s.get());
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) { reluTest(); }
};

#endif  // UTENSOR_NN_TESTS
