#ifndef UTENSOR_MATRIX_TEST 
#define UTENSOR_MATRIX_TEST 

#include "test.hpp"
#include "MatrixOps.hpp"
#include "tensorIdxImporter.hpp"

class matrixOpsTest : public Test {
    TensorIdxImporter t_import;
    Context ctx;

 public:
  void qMatMul(void) {

    testStart("Quantized Matrix Mul");

    ctx.gc();

    //inputs
    ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx"), "a");
    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx"), "a_min");
    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx"), "a_max");
    ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx"), "b");
    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx"), "b_min");
    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx"), "b_max");

    // reference outputs
    S_TENSOR c = ctx.add(t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx"), "c");
    S_TENSOR c_min = ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx"), "c_min");
    S_TENSOR c_max = ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx"), "c_max");


    //we need default constructor here
    //so we can get ride of the shapes here
    S_TENSOR out_c = ctx.add(new RamTensor<int>(c->getShape()), "out_c");
    S_TENSOR out_min = ctx.add(new RamTensor<float>(c_min->getShape()), "out_min");
    S_TENSOR out_max = ctx.add(new RamTensor<float>(c_max->getShape()), "out_max");

    //TList inputs = {a, a_min, a_max, b, b_min, b_max};
    //TList outputs = {out_c, out_min, out_max};

    //if you want tensors to be alive after .eval()
    //copies of the share_pointer needs to be here

    timer_start();
    //ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), inputs, outputs);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(),
         {"a", "a_min", "a_max", "b", "b_min", "b_max"},
         {"out_c", "out_min", "out_max"});
         
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<int>(c.get(), out_c.get()) + meanPercentErr<float>(c_min.get(), out_min.get()) +
                      meanPercentErr<float>(c_max.get(), out_max.get());

    passed(result == 0);
  }

  void runAll(void) { qMatMul(); }
};
#endif
