#ifndef UTENSOR_TEST
#define UTENSOR_TEST

#include <math.h>
#include <limits>
#include <string>
#include <vector>
#include "mbed.h"
#include "uTensor_util.hpp"

class Test {
 private:
  Timer timer;
  string testName;
  string summary;

 protected:
  void printStatus(string status) {
    int pLen = std::max(1, 30 - (int)testName.length());
    string msg = testName;

    for (int i = 0; i < pLen; i++) {
      msg += " ";
    }

    msg += "[ " + status + " ]";

    float lapsed_time = timer.read();
    if (lapsed_time != 0) {
      msg += " (" + std::to_string(lapsed_time * 1000) + " ms)";
    }

    summary += msg + "\r\n";

    if (print_test) printf("%s\r\n", msg.c_str());
  }

  void timer_start() { timer.start(); }

  void timer_stop() { timer.stop(); }

  void testStart(string _testName) {
    timer.reset();
    testName = _testName;
    numTotal++;
  }

  void passed(bool res = true) {
    timer.stop();

    if (!res) {
      failed();
      return;
    }

    if (testName == "") ERR_EXIT("Error: test name not cleared piror to test run\r\n");

    numOk++;
    printStatus("  OK  ");

    testName = "";
  }

  void failed() {
    timer.stop();

    if (testName == "") ERR_EXIT("Error: testStart is not called prior to test start\r\n");

    numFailed++;
    printStatus("** FAILED **");

    testName = "";
  }

  void warn() {
    timer.stop();

    if (testName == "") ERR_EXIT("Error: testStart is not called prior to test start\r\n");

    numWarn++;
    printStatus(" * WARN * ");

    testName = "";
  }

 public:
  unsigned int numOk;
  unsigned int numFailed;
  unsigned int numWarn;
  unsigned int numTotal;
  bool print_test;

  Test() {
    numOk = 0;
    numFailed = 0;
    numWarn = 0;
    testName = "";
    summary = "";
    print_test = false;
  }

  void printSummary(void) { printf("%s\r\n", summary.c_str()); }

  virtual void runAll(void) = 0;

  template <typename U>
  double sum(Tensor<U> input) {
    U* elem = input.getPointer({});
    double accm = 0.0;
    for (uint32_t i = 0; i < input.getSize(); i++) {
      accm += (double)elem[i];
    }

    return accm;
  }
  template <typename T>
  bool testshape(vector<T> src, vector<T> res, vector<uint8_t> permute) {
    bool pass = true;
    for (size_t i = 0; i < permute.size(); i++) {
      if (src[permute[i]] != res[i]) {
        pass = false;
        return pass;
      }
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

  template <typename U>
  static double meanAbsErr(Tensor<U> A, Tensor<U> B) {
    if (A.getSize() != B.getSize()) {
      ERR_EXIT("Test.meanAbsErr(): dimension mismatch\r\n");
    }

    U* elemA = A.getPointer({});
    U* elemB = B.getPointer({});

    double accm = 0.0;
    for (uint32_t i = 0; i < A.getSize(); i++) {
      accm += (double)fabs((float)elemB[i] - (float)elemA[i]);
    }

    return accm;
  }

  // A being the reference
  template <typename U>
  static double sumPercentErr(Tensor<U> A, Tensor<U> B) {
    if (A.getSize() != B.getSize()) {
      ERR_EXIT("Test.sumPercentErr(): dimension mismatch\r\n");
    }

    U* elemA = A.getPointer({});
    U* elemB = B.getPointer({});

    double accm = 0.0;
    for (uint32_t i = 0; i < A.getSize(); i++) {
      if (elemA[i] != 0.0f) {
        accm += (double)fabs(((float)elemB[i] - (float)elemA[i]) /
                             fabs((float)elemA[i]));
      } else {
        if (elemB[i] != 0) {
          accm += elemB[i];
        }
      }
    }
    return accm;
  }

  template <typename U>
  static double meanPercentErr(Tensor<U> A, Tensor<U> B) {
    double sum = sumPercentErr(A, B);
    return sum / A.getSize();
  }
};

// https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
void printBits(size_t const size, void const* const ptr) {
  unsigned char* b = (unsigned char*)ptr;
  unsigned char byte;
  int i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      printf("%d", byte);
    }
  }
  puts("");
}

#endif
