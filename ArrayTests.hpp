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
        TENSOR b = ctx.add(t_import.float_import ("/fs/testData/qB/in/Cast_1_0.idx"));
        TENSOR b_min = ctx.add(t_import.float_import("/fs/testData/qB/in/Min_1_0.idx"));
        TENSOR b_max = ctx.add(t_import.float_import("/fs/testData/qB/in/Max_1_0.idx"));

        //reference outputs
        TENSOR b_q_ref = ctx.add(t_import.ubyte_import("/fs/testData/qB/out/qB_0.idx"));
        TENSOR b_min_q_ref = ctx.add(t_import.float_import("/fs/testData/qB/out/qB_1.idx"));
        TENSOR b_max_q_ref = ctx.add(t_import.float_import("/fs/testData/qB/out/qb_2.idx"));

        TENSOR b_q = ctx.add(new RamTensor<unsigned char>(b_q_ref.lock()->getShape()));
        TENSOR b_min_q = ctx.add(new RamTensor<float>(b_min_q_ref.lock()->getShape()));
        TENSOR b_max_q = ctx.add(new RamTensor<float>(b_max_q_ref.lock()->getShape()));

        TList inputs = {b, b_min, b_max};
        TList outputs = {b_q, b_min_q, b_max_q};
        S_TENSOR out_b_q = b_q.lock();
        S_TENSOR out_b_min_q = b_min_q.lock();
        S_TENSOR out_b_max_q = b_max_q.lock();
        S_TENSOR ref_b_q = b_q_ref.lock();
        S_TENSOR ref_b_min_q = b_min_q_ref.lock();
        S_TENSOR ref_b_max_q = b_max_q_ref.lock();

        //Implementation goes here
        timer_start();
        ctx.push(new QuantizeV2Op(), inputs, outputs);
        ctx.eval();
        timer_stop();

        // printf("refMin is : %f \r\n", *(b_min_q_ref.getPointer({0})));
        // printf("outMin is : %f \r\n", *(b_min_q.getPointer({0})));
        // printf("diff : output(%f), outMin(%f), outMax(%f)\r\n", 
        //  meanPercentErr(b_q_ref, b_q), meanPercentErr(b_min_q_ref, b_min_q), meanPercentErr(b_max_q_ref, b_max_q));

        double result = meanPercentErr<unsigned char>(ref_b_q.get(), out_b_q.get()) + meanPercentErr<float>(ref_b_min_q.get(), out_b_min_q.get()) + meanPercentErr<float>(ref_b_max_q.get(), out_b_max_q.get());
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void dequantizeTest(void) {
        testStart("dequantize");

        //reference inputs
        TENSOR a = ctx.add(t_import.ubyte_import("/fs/testData/deQ/in/rQ_0.idx"));
        TENSOR a_min = ctx.add(t_import.float_import("/fs/testData/deQ/in/rQ_1.idx"));
        TENSOR a_max = ctx.add(t_import.float_import("/fs/testData/deQ/in/rQ_2.idx"));

        //reference outputs
        TENSOR out_ref = ctx.add(t_import.float_import("/fs/testData/deQ/out/deQ_0.idx"));

        //modify the checks below:
        TENSOR out = ctx.add(new RamTensor<float>(out_ref.lock()->getShape()));
        TList inputs = {a, a_min, a_max};
        TList outputs = {out};

        S_TENSOR out_val = out.lock();
        S_TENSOR ref_out = out_ref.lock();

        timer_start();
        ctx.push(new DequantizeOp(), inputs, outputs);
        ctx.eval();
        timer_stop();

        double result = meanPercentErr<float>(out_val.get(), ref_out.get());
        //passed(result < 0.0001);
        passed(result == 0);
    }

    void reshapeTest(void) {
        testStart("reshape");
        TensorIdxImporter t_import;

        //reference inputs
        TENSOR ref_a = ctx.add(t_import.float_import("/fs/testData/ref_reshape/in/Const_0.idx"));
        TENSOR ref_dim = ctx.add(t_import.int_import("/fs/testData/ref_reshape/in/Const_1_0.idx"));

        //reference outputs
        TENSOR out_ref = ctx.add(t_import.float_import("/fs/testData/ref_reshape/out/ref_reshape_0.idx"));

        //modify the checks below:
        TENSOR out = ctx.add(new RamTensor<float>(out_ref.lock()->getShape()));
        S_TENSOR out_val = out.lock();
        S_TENSOR ref_out = out_ref.lock();
        
        TList inputs = {ref_a, ref_dim};
        TList outputs = {out};

        timer_start();
        ctx.push(new ReshapeOp(), inputs, outputs);
        ctx.eval();
        timer_stop();

        double result = meanPercentErr<float>(out_val.get(), ref_out.get());
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
