#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"
#include "src/uTensor/ops/NnOps.hpp"

TensorIdxImporter t_import;
Context ctx;

void test_float_SoftmaxOp(void){
    ctx.gc();

    // input
    S_TENSOR logits = ctx.add(t_import.float_import("/fs/constants/Softmax/in/float_logits.idx"), "float");

    // output
    S_TENSOR out = ctx.add(new RamTensor<float>(logits->getShape()), "out");
    ctx.push(
        new SoftmaxOp<float, float>(),
        {"logits",},
        {"out",}
    );
    ctx.eval();

    Tensor* ref_output = t_import.float_import("/fs/constants/Softmax/out/ref_float_softmax.idx");
    double err = meanAbsErr<float>(out.get(), ref_output);
    EXPECT_EQ(EXPECT_EQ < 1e-6, true);
}