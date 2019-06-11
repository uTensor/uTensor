#include "test_helper.h"
Context ctx;
#include "models/qmul0.hpp"
#include "models/qmul1.hpp"
#include "models/qmul2.hpp"
#include "models/qmul3.hpp"
#include "models/qmul4.hpp"



void test_operators_qmul0(void) {
    ctx.gc();

    get_qmul0_ctx(ctx); 
    ctx.eval();
    S_TENSOR res = ctx.get({"c_0:0"});
    S_TENSOR ref = ctx.get({"ref_0:0"});

    double result = meanPercentErr<float>(ref.get(), res.get());
    EXPECT_LT(result, 0.1);

}

void test_operators_qmul1(void) {
    ctx.gc();

    get_qmul1_ctx(ctx); 
    ctx.eval();
    S_TENSOR res = ctx.get({"c_1:0"});
    S_TENSOR ref = ctx.get({"ref_1:0"});

    double result = meanPercentErr<float>(ref.get(), res.get());
    EXPECT_LT(result, 0.1);

}

void test_operators_qmul2(void) {
    ctx.gc();

    get_qmul2_ctx(ctx); 
    ctx.eval();
    S_TENSOR res = ctx.get({"c_2:0"});
    S_TENSOR ref = ctx.get({"ref_2:0"});

    double result = meanPercentErr<float>(ref.get(), res.get());
    EXPECT_LT(result, 0.1);

}

void test_operators_qmul3(void) {
    ctx.gc();

    get_qmul3_ctx(ctx); 
    ctx.eval();
    S_TENSOR res = ctx.get({"c_3:0"});
    S_TENSOR ref = ctx.get({"ref_3:0"});

    double result = meanPercentErr<float>(ref.get(), res.get());
    EXPECT_LT(result, 0.1);

}

void test_operators_qmul4(void) {
    ctx.gc();

    get_qmul4_ctx(ctx); 
    ctx.eval();
    S_TENSOR res = ctx.get({"c_4:0"});
    S_TENSOR ref = ctx.get({"ref_4:0"});

    double result = meanPercentErr<float>(ref.get(), res.get());
    EXPECT_LT(result, 0.1);

}


// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

UTENSOR_TEST(operators, qmul0, "Test qmul 0")
UTENSOR_TEST(operators, qmul1, "Test qmul 1")
UTENSOR_TEST(operators, qmul2, "Test qmul 2")
UTENSOR_TEST(operators, qmul3, "Test qmul 3")
UTENSOR_TEST(operators, qmul4, "Test qmul 4")


// Third, run like hell
UTENSOR_TEST_RUN()
