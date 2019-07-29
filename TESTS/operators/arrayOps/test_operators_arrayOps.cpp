#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"
#include "src/uTensor/ops/ArrayOps.hpp"

#include <iostream>
using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;


// Default to using GTest like asserts and expects as these give more info that unity
// We will forward these commands to unity in test_helper.h
void test_operators_quantizeV2(){
    S_TENSOR b_q_ref = ctx.addCached(hold(t_import.float_import ("/fs/constants/qB/in/Cast_1_0.idx")), "b_q_ref");
    S_TENSOR b_min_q_ref = ctx.addCached(hold(t_import.float_import("/fs/constants/qB/in/Min_1_0.idx")), "b_min_q_ref");
    S_TENSOR b_max_q_ref = ctx.addCached(hold(t_import.float_import("/fs/constants/qB/in/Max_1_0.idx")), "b_max_q_ref");

    //reference outputs
    S_TENSOR ref_b_q = ctx.addCached(hold(t_import.ubyte_import("/fs/constants/qB/out/qB_0.idx")), "ref_b_q");
    S_TENSOR ref_b_min_q = ctx.addCached(hold(t_import.float_import("/fs/constants/qB/out/qB_1.idx")), "ref_b_min_q");
    S_TENSOR ref_b_max_q = ctx.addCached(hold(t_import.float_import("/fs/constants/qB/out/qB_2.idx")), "ref_b_max_q");

    S_TENSOR out_b_q = ctx.addCached(hold(new RamTensor<unsigned char>(b_q_ref->getShape())), "b_q");
    S_TENSOR out_b_min_q = ctx.addCached(hold(new RamTensor<float>(b_min_q_ref->getShape())), "b_min_q");
    S_TENSOR out_b_max_q = ctx.addCached(hold(new RamTensor<float>(b_max_q_ref->getShape())), "b_max_q");

    //Implementation goes here
    ctx.push_static(hold(new QuantizeV2Op()), "QuantizeV2Op", {"b_q_ref", "b_min_q_ref", "b_max_q_ref"}, {"b_q", "b_min_q", "b_max_q"});
    ctx.eval();


    double result = meanPercentErr<unsigned char>(ref_b_q.get(), out_b_q.get()) + meanPercentErr<float>(ref_b_min_q.get(), out_b_min_q.get()) + meanPercentErr<float>(ref_b_max_q.get(), out_b_max_q.get());
    EXPECT_EQ(result, 0);
}

void test_operators_dequantize(void) {

    //reference inputs
    S_TENSOR a = ctx.addCached(hold(t_import.ubyte_import("/fs/constants/deQ/in/rQ_0.idx")), "a");
    S_TENSOR a_min = ctx.addCached(hold(t_import.float_import("/fs/constants/deQ/in/rQ_1.idx")), "a_min");
    S_TENSOR a_max = ctx.addCached(hold(t_import.float_import("/fs/constants/deQ/in/rQ_2.idx")), "a_max");

    //reference outputs
    S_TENSOR out_ref = ctx.addCached(hold(t_import.float_import("/fs/constants/deQ/out/deQ_0.idx")), "out_ref");

    //modify the checks below:
    S_TENSOR out = ctx.addCached(hold(new RamTensor<float>(out_ref->getShape())), "out");

    ctx.push_static(hold(new DequantizeOp()), "DequantizeOp", {"a", "a_min", "a_max"}, {"out"});
    ctx.eval();

    double result = meanPercentErr<float>(out.get(), out_ref.get());
    EXPECT_EQ(result, 0);
}

void test_operators_reshape(void) {
    //reference inputs
    S_TENSOR ref_a = ctx.addCached(hold(t_import.float_import("/fs/constants/ref_reshape/in/Const_0.idx")), "ref_a");
    S_TENSOR ref_dim = ctx.addCached(hold(t_import.int_import("/fs/constants/ref_reshape/in/Const_1_0.idx")), "ref_dim");

    //reference outputs
    S_TENSOR out_ref_2 = ctx.addCached(hold(t_import.float_import("/fs/constants/ref_reshape/out/ref_reshape_0.idx")), "out_ref_2");

    //modify the checks below:
    S_TENSOR out_2 = ctx.addCached(hold(new RamTensor<float>(out_ref_2->getShape())), "out_2");


    ctx.push_static(hold(new ReshapeOp()), "ReshapeOp", {"ref_a", "ref_dim"}, {"out_2"});
    ctx.eval();

    double result = meanPercentErr<float>(out_2.get(), out_ref_2.get());
    EXPECT_EQ(result, 0);
}

void test_operators_gather(void) {
    Tensor* input = new RamTensor<float>();
    Tensor* output = new RamTensor<float>();
    Tensor* out_ref = new RamTensor<float>();
    Tensor* indices  = new RamTensor<uint32_t>();
    TensorShape tmp({2, 2});
    TensorShape tmp2({3});

    input->init(tmp);
    output->init(tmp2);
    indices->init(tmp2);
    out_ref->init(tmp2);

    input->write<float>(0,0)[0] = 100.0;
    input->write<float>(0,0)[1] = 11.0;
    input->write<float>(0,0)[2] = 12.0;
    input->write<float>(0,0)[3] = 13.0;

    indices->write<uint32_t>(0,0)[0] = 1;
    indices->write<uint32_t>(0,0)[1] = 2;
    indices->write<uint32_t>(0,0)[2] = 1;

    out_ref->write<float>(0,0)[0] = 11.0;
    out_ref->write<float>(0,0)[1] = 12.0;
    out_ref->write<float>(0,0)[2] = 11.0;

    ctx.add(input, "g_input");
    ctx.add(output, "g_output");
    ctx.add(indices, "g_indices");

    ctx.push(new GatherOp<float>(),
            {"g_input", "g_indices", "g_indices"/*Not used*/},
            {"g_output"});
    ctx.eval();

    double result = meanPercentErr<float>(output, out_ref);
    EXPECT_EQ(result, 0);

}
// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(operators, quantizeV2, "Quantize V2 test")
UTENSOR_TEST(operators, dequantize, "Dequantization Test")
UTENSOR_TEST(operators, reshape, "Reshape Test")
UTENSOR_TEST(operators, gather, "Gather Test")


// Third, run like hell
UTENSOR_TEST_RUN()
