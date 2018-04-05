#ifndef UTENSOR_ARRAY_TESTS
#define UTENSOR_ARRAY_TESTS

#include "ArrayOps.hpp"
#include "test.hpp"
#include "tensorIdxImporter.hpp"
#include "context.hpp"
#include "tensor.hpp"

class ArrayOpsTest : public Test {
        TensorIdxImporter t_import;
        Context ctx;
public:
    void quantize_v2Test(void) {
        testStart("quantize_v2");

        //reference inputs  /Users/neitan01/Documents/mbed/uTensor.git/TESTS/scripts/PRE-GEN/qA
        S_TENSOR b_q_ref = ctx.addCached(hold(t_import.float_import ("/fs/testData/qB/in/Cast_1_0.idx")), "b_q_ref");
        S_TENSOR b_min_q_ref = ctx.addCached(hold(t_import.float_import("/fs/testData/qB/in/Min_1_0.idx")), "b_min_q_ref");
        S_TENSOR b_max_q_ref = ctx.addCached(hold(t_import.float_import("/fs/testData/qB/in/Max_1_0.idx")), "b_max_q_ref");

        //reference outputs
        S_TENSOR ref_b_q = ctx.addCached(hold(t_import.ubyte_import("/fs/testData/qB/out/qB_0.idx")), "ref_b_q");
        S_TENSOR ref_b_min_q = ctx.addCached(hold(t_import.float_import("/fs/testData/qB/out/qB_1.idx")), "ref_b_min_q");
        S_TENSOR ref_b_max_q = ctx.addCached(hold(t_import.float_import("/fs/testData/qB/out/qB_2.idx")), "ref_b_max_q");

        S_TENSOR out_b_q = ctx.addCached(hold(new RamTensor<unsigned char>(b_q_ref->getShape())), "b_q");
        S_TENSOR out_b_min_q = ctx.addCached(hold(new RamTensor<float>(b_min_q_ref->getShape())), "b_min_q");
        S_TENSOR out_b_max_q = ctx.addCached(hold(new RamTensor<float>(b_max_q_ref->getShape())), "b_max_q");

        //Implementation goes here
        timer_start();
        ctx.push_static(hold(new QuantizeV2Op()), "QuantizeV2Op", {"b_q_ref", "b_min_q_ref", "b_max_q_ref"}, {"b_q", "b_min_q", "b_max_q"});
        ctx.eval();
        timer_stop();


        double result = meanPercentErr<unsigned char>(ref_b_q.get(), out_b_q.get()) + meanPercentErr<float>(ref_b_min_q.get(), out_b_min_q.get()) + meanPercentErr<float>(ref_b_max_q.get(), out_b_max_q.get());
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void dequantizeTest(void) {
        testStart("dequantize");

        //reference inputs
        S_TENSOR a = ctx.addCached(hold(t_import.ubyte_import("/fs/testData/deQ/in/rQ_0.idx")), "a");
        S_TENSOR a_min = ctx.addCached(hold(t_import.float_import("/fs/testData/deQ/in/rQ_1.idx")), "a_min");
        S_TENSOR a_max = ctx.addCached(hold(t_import.float_import("/fs/testData/deQ/in/rQ_2.idx")), "a_max");

        //reference outputs
        S_TENSOR out_ref = ctx.addCached(hold(t_import.float_import("/fs/testData/deQ/out/deQ_0.idx")), "out_ref");

        //modify the checks below:
        S_TENSOR out = ctx.addCached(hold(new RamTensor<float>(out_ref->getShape())), "out");

        timer_start();
        ctx.push_static(hold(new DequantizeOp()), "DequantizeOp", {"a", "a_min", "a_max"}, {"out"});
        ctx.eval();
        timer_stop();

        double result = meanPercentErr<float>(out.get(), out_ref.get());
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void reshapeTest(void) {
        testStart("reshape");
        TensorIdxImporter t_import;

        //reference inputs
        S_TENSOR ref_a = ctx.addCached(hold(t_import.float_import("/fs/testData/ref_reshape/in/Const_0.idx")), "ref_a");
        S_TENSOR ref_dim = ctx.addCached(hold(t_import.int_import("/fs/testData/ref_reshape/in/Const_1_0.idx")), "ref_dim");

        //reference outputs
        S_TENSOR out_ref_2 = ctx.addCached(hold(t_import.float_import("/fs/testData/ref_reshape/out/ref_reshape_0.idx")), "out_ref_2");

        //modify the checks below:
        S_TENSOR out_2 = ctx.addCached(hold(new RamTensor<float>(out_ref_2->getShape())), "out_2");


        timer_start();
        ctx.push_static(hold(new ReshapeOp()), "ReshapeOp", {"ref_a", "ref_dim"}, {"out_2"});
        ctx.eval();
        timer_stop();

        double result = meanPercentErr<float>(out_2.get(), out_ref_2.get());
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void runAll(void) {
        quantize_v2Test();
        dequantizeTest();
        reshapeTest();
    }
};


#endif //UTENSOR_ARRAY_TESTS
