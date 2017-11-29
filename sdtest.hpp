#ifndef UTENSOR_SDTENSOR_TESTS
#define UTENSOR_SDTENSOR_TESTS

#include "test.hpp"
#include "sdtensor.hpp"
#include "tensorIdxImporter.hpp"

class SDTensorTest : public Test {
  public:
    void readTest(void) {
    testStart("sd import test");
    timer_start();
    Tensor* t = nullptr;
    {
    TensorIdxImporter t_import;
    t =
        t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx", "uchar1");
    }
    Tensor* s = new SDTensor<unsigned char>(t->getShape(), "sdf", "/fs/testData/idxImport/uint8_4d_power2.idx");
    const unsigned char* x = t->read<unsigned char>(0, 0);
    const unsigned char* y = s->read<unsigned char>(5, 5);
    timer_stop();
    passed(x[5] == y[0]);
    delete t;
    delete s;
  }
  void runAll() {
    readTest();
  }

};

#endif
