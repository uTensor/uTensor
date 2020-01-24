#include "test_helper.h"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/ops/NnOps.hpp"

TensorIdxImporter t_import;
Context ctx;

void test_operators_float_SoftmaxOp(void){
    ctx.gc();

    // input
    S_TENSOR logits = ctx.add(t_import.float_import("/fs/constants/Softmax/in/float_logits.idx"), "logits");

    // output
    S_TENSOR out = ctx.add(new RamTensor<float>(logits->getShape()), "out");
    ctx.push(
        new SoftmaxOp<float, float>(),
        {"logits",},
        {"out",}
    );
    ctx.eval();

    Tensor* ref_output = t_import.float_import("/fs/constants/Softmax/out/ref_float_softmax.idx");
    double err = meanAbsErr<float>(out, ref_output);
    EXPECT_EQ(err < 1e-6, true);
}

// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(operators, float_SoftmaxOp, "Test Softmax Op for float type")

// Third, run like hell
UTENSOR_TEST_RUN()
