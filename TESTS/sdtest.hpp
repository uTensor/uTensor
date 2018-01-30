#ifndef UTENSOR_SDTENSOR_TESTS
#define UTENSOR_SDTENSOR_TESTS

#include "test.hpp"
#include "sdtensor.hpp"
#include "tensorIdxImporter.hpp"

class SDTensorTest : public Test {
    TensorIdxImporter t_import;
    Context ctx;
  public:
    void rTest(void) {
    testStart("sd read test1");
    timer_start();
    Tensor* t = nullptr;
    t = t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx");
    Tensor* s = t_import.sd_ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx", 10);//the size of data is 10 elements

    const unsigned char* x = t->read<unsigned char>(0, 0);
    uint32_t x_res = 0;
    uint32_t y_res = 0;
    for (uint32_t index = 0; index < t->getSize(); index++) {
      const unsigned char* y = s->read<unsigned char>(index, 1);
      x_res += (uint32_t)x[index];
      y_res += (uint32_t)y[0];
    }

    passed(x_res == y_res);
    timer_stop();
    delete t;
    delete s;
  }
    void wTest(void) {
    testStart("sd write test2");
    timer_start();
    Tensor* h = t_import.sd_int_import("/fs/testData/qMatMul/res/wtest.idx", 5);//the size of data is 5 elements

    int* y = h->write<int>(0, 1);
    uint32_t res_x = 0;
    uint32_t res_y = 0;
    for (uint32_t i = 0; i < h->getSize(); i++) {
      y = h->write<int>(i, 1);
      y[0] = 's';
      unsigned char a = 's';
      res_x += (uint32_t)a;
    }

    for(uint32_t i = 0; i < h->getSize(); i++) {
      auto y = h->read<int>(i, 1);
      res_y += (uint32_t) *y;
    }

    passed(res_x == res_y);
    timer_stop();
    delete h;
  }
  /*  void wTest() {
    passed(x[5] == y[0]);
    testStart("sd read test2");
    y = s->read<unsigned char>(55, 5);
    passed(x[55] == y[0]);
    testStart("sd write1 test");
    unsigned char *y_w = s->write<unsigned char>(55, 5);
    unsigned char *x_w = t->write<unsigned char>(0, 0);
    y_w[0] = '5';
    x_w[55] = '5';
    }*/
  void qMatMul(void) {

    testStart("Quantized Matrix Mul");


    //inputs
//    ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx", "a"));
//    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx", "a_min"));
//    ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx", "a_max"));
    ctx.add(t_import.sd_ubyte_import("/fs/testData/qMatMul/in/qA_0.idx", 160), "a");
    ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/in/qA_1.idx", 1), "a_min");
    ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/in/qA_2.idx", 1), "a_max");
    ctx.add(t_import.sd_ubyte_import("/fs/testData/qMatMul/in/qB_0.idx", 128), "b");
    ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/in/qB_1.idx", 1), "b_min");
    ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/in/qB_2.idx", 1), "b_max");

    //ctx.add(t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx", "b"));
    //ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx", "b_min"));
    //ctx.add(t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx", "b_max"));
    // reference outputs
    S_TENSOR c = ctx.add(t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx"), "c");
    S_TENSOR c_min = ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx"), "c_min");
    S_TENSOR c_max = ctx.add(t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx"), "c_max");

    //we need default constructor here
    //so we can get ride of the shapes here
    S_TENSOR out_c = ctx.add(new SDTensor<int>(128), "out_c");
    S_TENSOR out_min = ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/res/min.idx", 1), "out_min");
    S_TENSOR out_max = ctx.add(t_import.sd_float_import("/fs/testData/qMatMul/res/max.idx", 1), "out_max");

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
  void runAll() {
    rTest();
    wTest();
    qMatMul();
  }

};

#endif
