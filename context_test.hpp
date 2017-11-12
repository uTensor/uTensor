#ifndef UTENSOR_CONTEXT_TESTS
#define UTENSOR_CONTEXT_TESTS

#include "mbed.h"
#include "uTensor_util.hpp"
#include "tensor.hpp"
#include "context.hpp"
#include "tensorIdxImporter.hpp"
#include "MatrixOps.hpp"
#include "MathOps.hpp"
#include "test.hpp"



class contextTest : public Test {

  TensorIdxImporter t_import;
  Context ctx;

public:

  void MatMalTest(void) {
    testStart("Context QntMatMal Op");
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
    ctx.push(new QntMatMulOp(), inputs, outputs);
    ctx.eval();
    timer_stop();

    double result = meanPercentErr<int>(ref_c_rptr.get(), out_c_rptr.get()) + meanPercentErr<float>(ref_min_rptr.get(), out_min_rptr.get()) +
                      meanPercentErr<float>(ref_max_rptr.get(), out_max_rptr.get());

    passed(result == 0);
  }

  void RefCountTest(void) {
    testStart("Context Ref Count");
    timer_start();
    //inputs
    TENSOR a = ctx.add(new RamTensor<int>({1,1,1}));
    TENSOR b = ctx.add(new RamTensor<int>({1,1,1}));
    TENSOR c = ctx.add(new RamTensor<int>({1,1,1}));

    //init values
    *(a.lock()->write<int>(0, 0)) = 1;
    *(b.lock()->write<int>(0, 0)) = 1;
    *(c.lock()->write<int>(0, 0)) = 1;

    // reference outputs
    TENSOR out = ctx.add(new RamTensor<int>({1,1,1}));
    S_TENSOR shr_out = out.lock();

    TList inputs0 = {a, b};
    TList outputs0 = {c};  //2
    ctx.push(new AddOp<int, int>(), inputs0, outputs0);

    TList inputs1 = {c, a};
    TList outputs1 = {b};  //3
    ctx.push(new AddOp<int, int>(), inputs1, outputs1);

    TList inputs2 = {a, b};
    TList outputs2 = {out};  //4
    ctx.push(new AddOp<int, int>(), inputs2, outputs2);
    ctx.eval();
    timer_stop();

    if(a.lock() || b.lock() || c.lock() || !out.lock()) {
        failed();
        return;
    }

    int result = *(shr_out->read<int>(0, 0));
    passed(result == 4);

  }


  void runAll(void) {
    MatMalTest();
    RefCountTest();
  }
};

#endif  // UTENSOR_IDX_IMPORTER_TESTS
