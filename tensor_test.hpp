#ifndef UTENSOR_impl_TESTS
#define UTENSOR_impl_TESTS

#include "test.hpp"
#include "NnOps.hpp"
#include "tensorIdxImporter.hpp"
#include <random>
#include <iostream>
#include <algorithm>

class read {
    public:
        void impl() {
          for (int i = 0; i < 1000000; i++) {
            i++;
          }
        }
};
class A {
  public:
      void impl() {
        for (int i = 0; i < 1000000; i++) {
          i++;
        }
      }
};

class B {
    public:
      virtual void impl() = 0;
};

class C : public B {
  public:
      void impl() override{
        for (int i = 0; i < 1000000; i++) {
          i++;
        }
      }
};
class D {
    private:
        read r;
  public:

      void impl() {
        r.impl();     
      }
};
template<typename T>
class vectesthelper {
    public:
        vectesthelper(size_t d1 = 0, size_t d2 = 0, size_t d3 = 0, T const & t = T()) :
            d1(d1), d2(d2), d3(d3), data(d1*d2*d3, t)
    {}

        T& operator() (size_t i, size_t j, size_t k) {
            return data[i *d2 * d3 + j * d3 + k];
        }
    std::vector<T> data;
    private:
        size_t d1, d2, d3;
};

class transTest : public Test {
  public:
    void a() {
        std::random_device rd;
        std::default_random_engine gen = std::default_random_engine(rd());

        for (int i = 0; i < 10; i++) {
        testStart("transtest");
        Tensor<int> inputTensor({10, 10, 100, 40});
        vector<uint32_t> g = inputTensor.getShape();
        vector<size_t> permute = {2, 3, 0, 1};
        std::shuffle(permute.begin(), permute.end(), gen);

        permuteIndexTransform trans(inputTensor.getShape(), permute);

        Tensor<int> output(trans.getNewShape());
        vector<uint32_t> s = output.getShape();
        bool res = testshape<uint32_t>(g, s, permute);
        passed(res);
        }
    }
    void b() {
        vector<int> in_d({2, 5, 4, 5, 2, 6, 5, 1, 3, 6, 7, 9, 1, 2, 3, 4, 3, 5, 6, 9, 2, 3, 3, 2});
     
        vector<int> ou_d({2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9, 1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2});

        Tensor<int> inputTensor({2, 3, 4});
        vector<size_t> permute = {0, 2, 1};

        permuteIndexTransform trans(inputTensor.getShape(), permute);
        size_t i = 15;
        size_t o = trans[i];
        testStart("start");
        bool res = testval(in_d[i], ou_d[o]);
        passed(res);

        res = false;

        testStart("star2");
        i = 5;
        o = trans[i];
        res = testval(in_d[i], ou_d[o]);
        passed(res);


        res = false;
        vector<int> in_d2({2, 2, 3, 5, 6, 6, 4, 5, 7, 5, 1, 9, 1, 3, 2, 2, 5, 3, 3, 6, 3, 4, 9, 2});
     
        vector<int> ou_d2({2, 1, 2, 3, 3, 2, 5, 2, 6, 5, 6, 3, 4, 3, 5, 6, 7, 3, 5, 4, 1, 9, 9, 2});

        Tensor<int> inputTensor2({2, 4, 3});
        vector<size_t> permute2 = {1, 2, 0};
        permuteIndexTransform trans2(inputTensor2.getShape(), permute2);
        for (uint32_t i = 0; i < 24; i++) {
          testStart("star3");
          o = trans2[i];
          res = testval(in_d2[i], ou_d2[o]);
          passed(res); 
          res = false;
        }

        res = false;
        testStart("star4");
        i = 11;
        o = trans2[i];
        res = testval(in_d2[i], ou_d2[o]);
        passed(res);
       

    }
    void runAll() {
        a();
        b();
    }
};
/*class tensorImplTest : public Test {
public:
    void aTest(void) {
        testStart("a");

        A *a = new A();
        timer_start();
        for (int i = 0; i < 100; i+=10) {
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        a->impl();
        }
        timer_stop();
        double result = 0.0;
        //passed(result < 0.0001);
        passed(result == 0);
    }
    void cTest() {
      testStart("b");
      B *c = new C();
      timer_start();
      for (int i = 0; i < 100; i+=10) {
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      c->impl();
      }
      timer_stop();
      double result = 0.0;
      passed(result == 0);
    }
    void dTest() {
      testStart("d");
      D *d = new D();
      timer_start();
      for (int i = 0; i < 100; i+=10) {
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      d->impl();
      }
      timer_stop();
      double result = 0.0;
      passed(result == 0);
    }

    void runAll(void) {
        aTest();
        cTest();
        dTest();
    }
};*/


#endif //UTENSOR_impl_TESTS
