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
    //inputs
    TENSOR a =
        ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx"));
    TENSOR a_min =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx"));
    TENSOR a_max =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx"));
    TENSOR b =
        ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx"));
    TENSOR b_min =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx"));
    TENSOR b_max =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx"));

    // reference outputs
    TENSOR c =
        ctx.add(t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx"));
    TENSOR c_min =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx"));
    TENSOR c_max =
        ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx"));


    //we need default constructor here
    //so we can get ride of the shapes here
    TENSOR out_c = ctx.add(new RamTensor<int>(c.lock()->getShape()));
    TENSOR out_min = ctx.add(new RamTensor<float>(c_min.lock()->getShape()));
    TENSOR out_max = ctx.add(new RamTensor<float>(c_max.lock()->getShape()));

    TList inputs = {a, a_min, a_max, b, b_min, b_max};
    TList outputs = {out_c, out_min, out_max};

    //if you want tensors to be alive after .eval()
    //copies of the share_pointer needs to be here
    S_TENSOR ref_c_rptr = c.lock();
    S_TENSOR ref_min_rptr = c_min.lock();
    S_TENSOR ref_max_rptr = c_max.lock();
    S_TENSOR out_c_rptr = out_c.lock();
    S_TENSOR out_min_rptr = out_min.lock();
    S_TENSOR out_max_rptr = out_max.lock();
    

    timer_start();
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<int>(ref_c_rptr.get(), out_c_rptr.get()) + meanPercentErr<float>(ref_min_rptr.get(), out_min_rptr.get()) +
                      meanPercentErr<float>(ref_max_rptr.get(), out_max_rptr.get());

    passed(result == 0);
  }

  void runAll(void) { qMatMul(); }
};
#endif
