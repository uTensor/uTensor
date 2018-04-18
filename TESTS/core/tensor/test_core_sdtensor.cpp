#include "test_helper.h"
#include "uTensor/loaders/tensorIdxImporter.hpp"

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

// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(core, readSDTensor, "Test sdtensor read interface")


// Third, run like hell
UTENSOR_TEST_RUN()

