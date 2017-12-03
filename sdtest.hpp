#ifndef UTENSOR_SDTENSOR_TESTS
#define UTENSOR_SDTENSOR_TESTS

#include "test.hpp"
#include "sdtensor.hpp"
#include "tensorIdxImporter.hpp"

class SDTensorTest : public Test {
  public:
    void rwTest(void) {
    testStart("sd read test1");
    timer_start();
    Tensor* t = nullptr;
    {
    TensorIdxImporter t_import;
    t =
        t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx", "uchar1");
    }
    Tensor* s = new SDTensor<unsigned char>(t->getShape(), "sdf", "/fs/testData/idxImport/uint8_4d_power2.idx", 50);//the size of data is 50 elements

    const unsigned char* x = t->read<unsigned char>(0, 0);
    const unsigned char* y = s->read<unsigned char>(5, 5);
    passed(x[5] == y[0]);
    testStart("sd read test2");
    y = s->read<unsigned char>(55, 5);
    passed(x[55] == y[0]);
    testStart("sd write1 test");
    unsigned char *y_w = s->write<unsigned char>(55, 5);
    unsigned char *x_w = t->write<unsigned char>(0, 0);
    y_w[0] = '5';
    x_w[55] = '5';
    passed(x_w[55] + x_w[56] == y_w[0] + y_w[1]);
    timer_stop();
    delete t;
    delete s;
  }
  void runAll() {
    rwTest();
  }

};

#endif
