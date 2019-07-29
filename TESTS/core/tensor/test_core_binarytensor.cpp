#include "test_helper.h"
#include "src/uTensor/loaders/tensorIdxImporter.hpp"

#include <iostream>
#include <algorithm>
#include <random>

using std::cout;
using std::endl;

TensorIdxImporter t_import;
Context ctx;
const int val[18] = {3, 5, 6, 7, 8, 9, 10, 21, 9, 32,
                1, 2, 9, 3, 8, 67, 8, 9};
void test_core_resizeBinaryTensor() {
    Tensor* a = new BinaryTensor<int>({3, 2, 3}, val);
    TensorShape v({1, 5, 8});
    a->resize(v);
    bool res = testsize(1 * 5 * 8, a->getSize());
    EXPECT_EQ(res, true); 
    delete a;
}

void test_core_reshapeBinaryTensor() {
  bool res = false;

  for (int i = 0; i < 9; i++) {
    std::default_random_engine gen;
    TensorShape tmp({3, 2, 3});
    std::string a_s = "input" + to_string(i);
    S_TENSOR inputTensor = ctx.add(new BinaryTensor<int>(tmp, val), a_s.c_str());
    vector<uint8_t> permute = {2, 1, 0};
    TensorShape g = inputTensor->getShape();
    std::shuffle(permute.begin(), permute.end(), gen);

    permuteIndexTransform trans(inputTensor->getShape(), permute);

    std::string a_o = "output" + to_string(i);
    S_TENSOR output = ctx.add(new BinaryTensor<int>(trans.getNewShape(), val), a_o.c_str());
    TensorShape s = output->getShape();
    res = testshape<uint32_t>(g, s, permute);
    if (!res) {
      break;
    }
  }
  EXPECT_EQ(res, true);
}

// source numpy array dim (2, 3, 4)
const int input_1[24] = {2, 5, 4, 5, 2, 6, 5, 1, 3, 6, 7, 9,
                       1, 2, 3, 4, 3, 5, 6, 9, 2, 3, 3, 2};

const int input_2[24] = {2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9,
                       1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2};

const int input_3[27] = {8, 6, 0, 1, 3, 9, 4, 7, 3, 2, 0, 4, 0, 9,
                       0, 6, 0, 6, 8, 6, 8, 3, 2, 4, 2, 7, 8};
// Should we break this into 3 tests?
void test_core_permuteBinaryTensor() {

  // target numpy array (after np.transpose(0, 2, 1))
  vector<int> output_1({2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9,
                        1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2});

  S_TENSOR inputTensor2 = ctx.add(new BinaryTensor<int>({2, 3, 4}, input_1), "inputTensor2");
  vector<uint8_t> permute = {0, 2, 1};

  permuteIndexTransform trans(inputTensor2->getShape(), permute);
  size_t out_index = 0;
  bool res = false;

  for (uint32_t i = 0; i < 24; i++) {
    out_index = trans[i];
    res = testval(input_1[i], output_1[out_index]);
    if (!res) {
      break;
    }
  }
  EXPECT_EQ(res, true);

  res = false;

  // target numpy arrar (after transpose(1, 2, 0))
  vector<int> output_2({2, 1, 2, 3, 3, 2, 5, 2, 6, 5, 6, 3,
                        4, 3, 5, 6, 7, 3, 5, 4, 1, 9, 9, 2});

  S_TENSOR inputTensor3 = ctx.add(new BinaryTensor<int>({2, 4, 3}, input_2), "inputTensor3");
  vector<uint8_t> permute2 = {1, 2, 0};
  permuteIndexTransform trans2(inputTensor3->getShape(), permute2);
  for (uint32_t i = 0; i < 24; i++) {
    out_index = trans2[i];
    res = testval(input_2[i], output_2[out_index]);
    if (!res) {
      break;
    }
  }
  EXPECT_EQ(res, true);
  res = false;


  vector<int> output_3({8, 2, 8, 1, 0, 3, 4, 6, 2, 6, 0, 6, 3, 9,
                        2, 7, 0, 7, 0, 4, 8, 9, 0, 4, 3, 6, 8});

  S_TENSOR inputTensor4 = ctx.add(new BinaryTensor<int>({1, 3, 3, 3}, input_3), "inputTensor4");
  vector<uint8_t> permute3 = {0, 3, 2, 1};
  permuteIndexTransform trans3(inputTensor4->getShape(), permute3);
  for (uint32_t i = 0; i < 27; i++) {
    out_index = trans3[i];
    res = testval(input_3[i], output_3[out_index]);
    if (!res) {
      break;
    }
  }
  EXPECT_EQ(res, true);
}


// First configure the uTensor test runner
UTENSOR_TEST_CONFIGURE()

// Second declare tests to run
UTENSOR_TEST(core, resizeBinaryTensor, "Test resize tensor")
UTENSOR_TEST(core, reshapeBinaryTensor, "Test reshape tensor")
UTENSOR_TEST(core, permuteBinaryTensor, "Test permute tensor")


// Third, run like hell
UTENSOR_TEST_RUN()

