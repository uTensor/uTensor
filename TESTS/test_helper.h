#ifndef __TEST_HELPER_H__
#define __TEST_HELPER_H__
#include "unity.h"
#include "src/uTensor/core/context.hpp"

#ifdef COMPILE_GREENTEA
// Handle all the boilerplate code
#include "mbed.h"	
#include "greentea-client/test_env.h"	
#include "unity.h"	
#include "utest.h"	
#include <vector>
#include "FATFileSystem.h"
#include "SDBlockDevice.h"

using namespace utest::v1;

// Custom setup handler required for proper Greentea support
utest::v1::status_t greentea_setup(const size_t number_of_cases) {
    //Timeout 20
    GREENTEA_SETUP(20, "default_auto");
    // Call the default reporting function
    return greentea_test_setup_handler(number_of_cases);
}

#define EXPECT_EQ(x, y) TEST_ASSERT( (x) == y )

#define UTENSOR_TEST_CONFIGURE() std::vector<Case> cases(); \
    Serial pc(USBTX, USBRX, 115200); \
+SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO, MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS); \
+FATFileSystem fs("fs");

#define UTENSOR_TEST(x, y, message) cases.push_back(Case( message, test_ ## x ## _ ## y ));

#define UTENSOR_TEST_RUN() Specification specification(greentea_setup, cases.data); \
    int main(){ \
        ON_ERR(bd.init(), "SDBlockDevice init "); \
        ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". "); \
        return Harness::run(specification); \
    }


#else /* GTest */

#include "gtest/gtest.h"
#define UTENSOR_TEST_CONFIGURE() /* pass */

#define UTENSOR_TEST(x, y, message) GTEST_TEST(x, y){ test_ ## x ## _ ## y(); }

// Google does this better than I can
#define UTENSOR_TEST_RUN() int main(int argc, char** argv) { \
        ::testing::InitGoogleTest(&argc, argv); \
        auto out = RUN_ALL_TESTS(); \
        return out; \
    }

#endif

// Math functions go here TODO move me
//
template <typename U>
double meanAbsErr(Tensor* A, Tensor* B) {
  uint32_t size_A, size_B;
  size_A = A->getSize();
  size_B = B->getSize();
  if (A->getSize() != B->getSize()) {
    /* %lu is different for 64bit machines. Need to make this cross platform
    TensorShape shape_A, shape_B;
    shape_A = A->getShape();
    shape_B = B->getShape();
    printf("shape_A: ");
    for (auto s : shape_A) {
      printf("%lu, ", s);
    }
    printf("\n");
    printf("shape_B: ");
    for (auto s : shape_B) {
      printf("%lu, ", s);
    }
    printf("\n");
    printf("size A: %lu, size B: %lu\r\n", size_A, size_B);
    */
    ERR_EXIT("Test.meanAbsErr(): dimension mismatch\r\n");
  }

  const U* elemA = A->read<U>(0, 0);
  const U* elemB = B->read<U>(0, 0);

  double accm_err = 0.0;
  double total_size = (double) A->getSize();
  for (uint32_t i = 0; i < A->getSize(); i++) {
    accm_err += ((double)fabs((float)elemB[i] - (float)elemA[i]))/ total_size;
  }

  return accm_err;
}

// A being the reference
template <typename U>
double sumPercentErr(Tensor* A, Tensor* B) {
  uint32_t size_A, size_B;
  size_A = A->getSize();
  size_B = B->getSize();
  if (A->getSize() != B->getSize()) {
    
    ERR_EXIT("Test.sumPercentErr(): dimension mismatch\r\n");
  }


  double accm = 0.0;
  for (uint32_t i = 0; i < A->getSize(); i++) {
  const U* elemA = A->read<U>(i, 1);
  const U* elemB = B->read<U>(i, 1);
    if (elemA[0] != 0.0f) {
      accm += (double)fabs(((float)elemB[0] - (float)elemA[0]) / ((float)elemA[0]));
    } else {
      if (elemB[0] != 0) {
        accm += std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return accm;
}
template<typename U>
double meanPercentErr(Tensor* A, Tensor* B) {
  double sum = sumPercentErr<U>(A, B);
  return sum / A->getSize();
}

template<typename U>
double sum(Tensor* input) {
  const U* elem = input->read<U>(0, 0);
  double accm = 0.0;
  for (uint32_t i = 0; i < input->getSize(); i++) {
    accm += (double)elem[i];
  }

  return accm;
}
template <typename T>
bool testshape(std::vector<T> src, std::vector<T> res, std::vector<uint8_t> permute) {
  bool pass = true;
  for (size_t i = 0; i < permute.size(); i++) {
    if (src[permute[i]] != res[i]) {
      pass = false;
      return pass;
    }
  }
  return pass;
}

bool testsize(uint32_t src, uint32_t res) {
  bool pass = true;
  if (src != res) {
      pass = false;
      return pass;
  }
  return pass;
}
template <typename T>
bool testval(T src, T res) {
  bool pass = true;
  if (src == res) {
    return pass;
  }
  return false;
}

#endif
