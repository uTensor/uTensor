#ifndef UTENSOR_CONTEXT_TESTS
#define UTENSOR_CONTEXT_TESTS

#include "mbed.h"
#include "uTensor_util.hpp"
#include "tensor.hpp"
#include "context.hpp"
#include "tensorIdxImporter.hpp"
#include "MatrixOps.hpp"
#include "test.hpp"



class contextTest : public Test {

  TensorIdxImporter t_import;

public:

  void MatMalTest(void) {
    testStart("Context QntMatMal Op");
    //inputs
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


    Tensor* out_c = new RamTensor<int>(c->getShape());
    Tensor* out_min = new RamTensor<float>(c_min->getShape());
    Tensor* out_max = new RamTensor<float>(c_max->getShape());

    TList inputs = {a, a_min, a_max, b, b_min, b_max};
    TList outputs = {out_c, out_min, out_max};
    Operator* matMal = new QntMatMulOp();

    Context ctx;
    timer_start();
    ctx.push(matMal, inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<int>(c, out_c) + meanPercentErr<float>(c_min, out_min) +
                      meanPercentErr<float>(c_max, out_max);

    passed(result == 0);
  }

  void runAll(void) {
    MatMalTest();
  }
};

#endif  // UTENSOR_IDX_IMPORTER_TESTS