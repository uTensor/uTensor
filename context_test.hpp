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

private:

  void codeGenStatfulHelper(TName state) {
    ctx.add(TensorConstant<uint32_t>({1}, 1), "incr_val");
    ctx.add(new RamTensor<uint32_t>({1}), "out");  //gc problem?
    ctx.push(new AddOp<uint32_t, uint32_t>(), {"incr_val", state}, {"out"});
    ctx.eval();
  }

public:
  void RefCountTest(void) {
    testStart("Context Ref Count");
    
    ctx.gc();

    timer_start();
    //inputs
    
    S_TENSOR a = ctx.add(new RamTensor<int>({1,1,1}), "a");
    S_TENSOR b = ctx.add(new RamTensor<int>({1,1,1}), "b");
    S_TENSOR c = ctx.add(new RamTensor<int>({1,1,1}), "c");

    //init values
    *(a->write<int>(0, 0)) = 1;
    *(b->write<int>(0, 0)) = 1;
    *(c->write<int>(0, 0)) = 1;

    // reference outputs
    S_TENSOR out = ctx.add(new RamTensor<int>({1,1,1}), "out");

    TNameList inputs0 = {"a", "b"};
    TNameList outputs0 = {"c"};  //2
    ctx.push(new AddOp<int, int>(), inputs0, outputs0);

    TNameList inputs1 = {"c", "a"};
    TNameList outputs1 = {"b"};  //3
    ctx.push(new AddOp<int, int>(), inputs1, outputs1);

    TNameList inputs2 = {"a", "b"};
    TNameList outputs2 = {"out"};  //4
    ctx.push(new AddOp<int, int>(), inputs2, outputs2);
    ctx.eval();
    timer_stop();

    if(a.use_count() != 1 || b.use_count() != 1  || c.use_count() != 1  || out.use_count() != 2) {
        failed();
        return;
    }

    int result = *(out->read<int>(0, 0));
    passed(result == 4);

  }

  void codeGenTemplate(void) {
    testStart("codeGenTemplate");
    ctx.gc();
    S_TENSOR out;
    for(auto i = 0; i < 5; i++) {
      S_TENSOR state = ctx.add_static(hold(TensorConstant<uint32_t>({1}, 0)), "state");
      codeGenStatfulHelper("state");
      out = ctx.get("out");
      *(state->write<uint32_t>(0, 0)) = *(out->read<uint32_t>(0, 0));
      ctx.gc();
    }

    int result = *(out->read<int>(0, 0));
    passed(result == 5);

  }


  void runAll(void) {
    RefCountTest();
    codeGenTemplate();
  }
};

#endif  // UTENSOR_IDX_IMPORTER_TESTS
