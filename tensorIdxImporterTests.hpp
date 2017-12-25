#ifndef UTENSOR_IDX_IMPORTER_TESTS
#define UTENSOR_IDX_IMPORTER_TESTS

#include "tensorIdxImporter.hpp"
#include "test.hpp"

class idxImporterTest : public Test {
 public:
  void ntoh32Test(void) {
    testStart("ntoh32 test");
    uint32_t input = 63;
    timer_start();
    uint32_t result = ntoh32(input);
    timer_stop();
    passed(result == 1056964608);
  }

  void ucharTest(void) {
    testStart("uchar import test");
    TensorIdxImporter t_import;
    timer_start();
    Tensor* t =
        t_import.ubyte_import("/fs/testData/idxImport/uint8_4d_power2.idx");
    timer_stop();
    double result = sum<unsigned char>(t);
    passed(result == 4518);
    delete t;
  }

  void shortTest(void) {
    testStart("short import test");
    TensorIdxImporter t_import;
    timer_start();
    Tensor* t =
        t_import.short_import("/fs/testData/idxImport/int16_4d_power2.idx");
    timer_stop();
    double result = sum<short>(t);
    passed(result == 270250);
    delete t;
  }

  void intTest(void) {
    testStart("int import test");
    TensorIdxImporter t_import;
    timer_start();
    Tensor* t =
        t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    timer_stop();
    double result = sum<int>(t);
    passed(result == 7158278745);
    delete t;
  }

  void floatTest(void) {
    testStart("float import test");
    TensorIdxImporter t_import;
    timer_start();
    Tensor* t =
        t_import.float_import("/fs/testData/idxImport/float_4d_power2.idx");
    timer_stop();

    double result = sum<float>(t);

    DEBUG("***floating point test yielded: %.8e\r\n", (float)result);
    passed((float)result == -1.0f);
    delete t;
  }

  void runAll(void) {
    ntoh32Test();
    ucharTest();
    shortTest();
    intTest();
    floatTest();
  }
};

#endif  // UTENSOR_IDX_IMPORTER_TESTS
