#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"
#include "MatrixOps.hpp"

#include <iostream>
using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;

double alpha = 0.0001;

void test_operators_fusedConvMaxpool12(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size2_input.idx"), "x_1_2");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size2_weights.idx"), "w_1_2");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth1_pool_size2_output.idx"), "ref_1_2");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_1_2");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 2, 2, 1 },SAME),
            { "x_1_2", "w_1_2"}, {"out_1_2"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool13(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size3_input.idx"), "x_1_3");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size3_weights.idx"), "w_1_3");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth1_pool_size3_output.idx"), "ref_1_3");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_1_3");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 3, 3, 1 },SAME),
            { "x_1_3", "w_1_3"}, {"out_1_3"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool14(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size4_input.idx"), "x_1_4");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth1_pool_size4_weights.idx"), "w_1_4");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth1_pool_size4_output.idx"), "ref_1_4");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_1_4");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 4, 4, 1 },SAME),
            { "x_1_4", "w_1_4"}, {"out_1_4"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool22(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size2_input.idx"), "x_2_2");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size2_weights.idx"), "w_2_2");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth2_pool_size2_output.idx"), "ref_2_2");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_2_2");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 2, 2, 1 },SAME),
            { "x_2_2", "w_2_2"}, {"out_2_2"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool23(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size3_input.idx"), "x_2_3");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size3_weights.idx"), "w_2_3");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth2_pool_size3_output.idx"), "ref_2_3");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_2_3");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 3, 3, 1 },SAME),
            { "x_2_3", "w_2_3"}, {"out_2_3"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool24(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size4_input.idx"), "x_2_4");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth2_pool_size4_weights.idx"), "w_2_4");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth2_pool_size4_output.idx"), "ref_2_4");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_2_4");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 4, 4, 1 },SAME),
            { "x_2_4", "w_2_4"}, {"out_2_4"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool32(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size2_input.idx"), "x_3_2");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size2_weights.idx"), "w_3_2");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth3_pool_size2_output.idx"), "ref_3_2");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_3_2");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 2, 2, 1 },SAME),
            { "x_3_2", "w_3_2"}, {"out_3_2"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool33(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size3_input.idx"), "x_3_3");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size3_weights.idx"), "w_3_3");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth3_pool_size3_output.idx"), "ref_3_3");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_3_3");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 3, 3, 1 },SAME),
            { "x_3_3", "w_3_3"}, {"out_3_3"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    EXPECT_EQ(result < alpha, true);

}
void test_operators_fusedConvMaxpool34(void) {
    ctx.gc();

    //inputs
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size4_input.idx"), "x_3_4");
    ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/in/depth3_pool_size4_weights.idx"), "w_3_4");

    //reference outputs
    S_TENSOR ref = ctx.add(t_import.float_import("/fs/constants/fusedConvMaxpool/out/depth3_pool_size4_output.idx"), "ref_3_4");

    //Outputs
    S_TENSOR out = ctx.add(new RamTensor<float>(ref->getShape()), "out_3_4");

    ctx.push(new FusedConvMaxpoolOp<float,float,float>({ 1, 1 }, { 1, 4, 4, 1 },SAME),
            { "x_3_4", "w_3_4"}, {"out_3_4"});

    ctx.eval();

    double result = meanPercentErr<float>(ref.get(), out.get());
    //printf("Mean Percent Error %f", result);
    EXPECT_EQ(result < alpha, true);

}
 
// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(operators, fusedConvMaxpool12, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool13, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool14, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool22, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool23, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool24, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool32, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool33, "Generated Fused Conv Maxpool test")
UTENSOR_TEST(operators, fusedConvMaxpool34, "Generated Fused Conv Maxpool test")


// Third, run like hell
UTENSOR_TEST_RUN()

