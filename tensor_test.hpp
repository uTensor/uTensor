#ifndef UTENSOR_impl_TESTS
#define UTENSOR_impl_TESTS

#include <algorithm>
#include <random>
#include "NnOps.hpp"
#include "tensorIdxImporter.hpp"
#include "test.hpp"


class tensorTest : public Test {
  public:
      void runResize() {
      testStart("tensortest");
          Tensor* a = new RamTensor<int>({3, 2, 3});
          std::vector<uint32_t> v({1, 5, 8});
          a->resize(v);
          bool res = testsize(1 * 5 * 8, a->getSize());
         passed(res); 
         delete a;
      }

      void runAll() {
          runResize();
      }
};

class transTest : public Test {
  Context ctx;
 public:
  void runShapeTest() {
    bool res = false;

    testStart("transtest");
    for (int i = 0; i < 9; i++) {
      std::default_random_engine gen;
      vector<uint32_t> tmp({2, 3, 4, 5});
      std::string a_s = "input" + std::to_string(i);
      S_TENSOR inputTensor = ctx.add(new RamTensor<int>(tmp), a_s);
      vector<uint8_t> permute = {2, 3, 1, 0};
      vector<uint32_t> g = inputTensor->getShape();
      std::shuffle(permute.begin(), permute.end(), gen);

      permuteIndexTransform trans(inputTensor->getShape(), permute);

      std::string a_o = "output" + std::to_string(i);
      S_TENSOR output = ctx.add(new RamTensor<int>(trans.getNewShape()), a_o);
      vector<uint32_t> s = output->getShape();
      res = testshape<uint32_t>(g, s, permute);
      if (!res) {
        break;
      }
    }
    passed(res);
  }
  void runPermuteTest() {
    // source numpy array dim (2, 3, 4)
    vector<int> input_1({2, 5, 4, 5, 2, 6, 5, 1, 3, 6, 7, 9,
                         1, 2, 3, 4, 3, 5, 6, 9, 2, 3, 3, 2});
    // target numpy array (after np.transpose(0, 2, 1))
    vector<int> output_1({2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9,
                          1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2});

    S_TENSOR inputTensor2 = ctx.add(new RamTensor<int>({2, 3, 4}), "inputTensor2");
    vector<uint8_t> permute = {0, 2, 1};

    permuteIndexTransform trans(inputTensor2->getShape(), permute);
    size_t out_index = 0;
    bool res = false;

    testStart("test vec 1 for transform");
    for (uint32_t i = 0; i < input_1.size(); i++) {
      out_index = trans[i];
      res = testval(input_1[i], output_1[out_index]);
      if (!res) {
        break;
      }
    }
    passed(res);

    res = false;
    // source numpy array dim (2, 4, 3)
    vector<int> input_2({2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9,
                         1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2});

    // target numpy arrar (after transpose(1, 2, 0))
    vector<int> output_2({2, 1, 2, 3, 3, 2, 5, 2, 6, 5, 6, 3,
                          4, 3, 5, 6, 7, 3, 5, 4, 1, 9, 9, 2});

    S_TENSOR inputTensor3 = ctx.add(new RamTensor<int>({2, 4, 3}), "inputTensor3");
    vector<uint8_t> permute2 = {1, 2, 0};
    permuteIndexTransform trans2(inputTensor3->getShape(), permute2);
    testStart("test vec 2 for transform");
    for (uint32_t i = 0; i < input_2.size(); i++) {
      out_index = trans2[i];
      res = testval(input_2[i], output_2[out_index]);
      if (!res) {
        break;
      }
    }
    passed(res);
    res = false;

    vector<int> input_3({8, 6, 0, 1, 3, 9, 4, 7, 3, 2, 0, 4, 0, 9,
                         0, 6, 0, 6, 8, 6, 8, 3, 2, 4, 2, 7, 8});

    vector<int> output_3({8, 2, 8, 1, 0, 3, 4, 6, 2, 6, 0, 6, 3, 9,
                          2, 7, 0, 7, 0, 4, 8, 9, 0, 4, 3, 6, 8});

    S_TENSOR inputTensor4 = ctx.add(new RamTensor<int>({1, 3, 3, 3}), "inputTensor4");
    vector<uint8_t> permute3 = {0, 3, 2, 1};
    permuteIndexTransform trans3(inputTensor4->getShape(), permute3);
    testStart("test vec 4d for transform");
    for (uint32_t i = 0; i < input_3.size(); i++) {
      out_index = trans3[i];
      res = testval(input_3[i], output_3[out_index]);
      if (!res) {
        break;
      }
    }
    passed(res);
  }
  void runAll() {
    runPermuteTest();
    runShapeTest();
  }
};

#endif  // UTENSOR_impl_TESTS
