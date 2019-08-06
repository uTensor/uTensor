#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"
#include "MatrixOps.hpp"

#include <iostream>
#include <algorithm>
#include <random>

using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;

void test_core_readSDTensor() {
    Tensor* t = nullptr;
    t = t_import.ubyte_import("/fs/constants/idxImport/uint8_4d_power2.idx");
    Tensor* s = t_import.sd_ubyte_import("/fs/constants/idxImport/uint8_4d_power2.idx", 10);//the size of data is 10 elements

    const unsigned char* x = t->read<unsigned char>(0, 0);
    uint32_t x_res = 0;
    uint32_t y_res = 0;
    for (uint32_t index = 0; index < t->getSize(); index++) {
      const unsigned char* y = s->read<unsigned char>(index, 1);
      x_res += (uint32_t)x[index];
      y_res += (uint32_t)y[0];
    }

    EXPECT_EQ(x_res, y_res);
    delete t;
    delete s;
}


void test_core_writeSDTensor() {

    Tensor* h = t_import.sd_int_import("/fs/constants/qMatMul/sdstore/wtest.idx", 81);//the size of data is 5 elements


    int res_x = 0;
    int res_y = 0;
    for (uint32_t i = 0; i < h->getSize(); i++) {
      int* y = h->write<int>(i, 1);
      y[0] = 2;
      int a = 2;
      res_x += a;
    }

    for(uint32_t i = 0; i < h->getSize(); i++) {
      auto y = h->read<int>(i, 1);
      res_y += (int) *y; 
    }

    EXPECT_EQ(res_x, res_y);
    delete h;
}

void test_core_matmulSDtensor() {
    ctx.add(t_import.sd_ubyte_import("/fs/constants/qMatMul/in/qA_0.idx", 160), "a");
    ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/in/qA_1.idx", 1), "a_min");
    ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/in/qA_2.idx", 1), "a_max");
    ctx.add(t_import.sd_ubyte_import("/fs/constants/qMatMul/in/qB_0.idx", 128), "b");
    ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/in/qB_1.idx", 1), "b_min");
    ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/in/qB_2.idx", 1), "b_max");

    // reference outputs
    S_TENSOR c = ctx.add(t_import.int_import("/fs/constants/qMatMul/out/qMatMul_0.idx"), "c");
    S_TENSOR c_min = ctx.add(t_import.float_import("/fs/constants/qMatMul/out/qMatMul_1.idx"), "c_min");
    S_TENSOR c_max = ctx.add(t_import.float_import("/fs/constants/qMatMul/out/qMatMul_2.idx"), "c_max");

    //we need default constructor here
    //so we can get ride of the shapes here
    S_TENSOR out_c = ctx.add(new SDTensor<int>(128), "out_c");
    S_TENSOR out_min = ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/sdstore/min.idx", 1), "out_min");
    S_TENSOR out_max = ctx.add(t_import.sd_float_import("/fs/constants/qMatMul/sdstore/max.idx", 1), "out_max");



    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(),
         {"a", "a_min", "a_max", "b", "b_min", "b_max"},
         {"out_c", "out_min", "out_max"});
         
    ctx.eval();

    double result = meanPercentErr<int>(c.get(), out_c.get()) + meanPercentErr<float>(c_min.get(), out_min.get()) +
                      meanPercentErr<float>(c_max.get(), out_max.get());

    EXPECT_EQ(result, 0);
}
// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(core, readSDTensor, "Test sdtensor read interface")
UTENSOR_TEST(core, writeSDTensor, "Test sdtensor write interface")
UTENSOR_TEST(core, matmulSDtensor, "Test sdtensor matrix mul test")


// Third, run like hell
UTENSOR_TEST_RUN()

