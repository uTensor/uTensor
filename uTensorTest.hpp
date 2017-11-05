#ifndef UTENSORTEST_HPP
#define UTENSORTEST_HPP

#include <math.h>
#include <limits>
#include <string>
#include <vector>
#include "mbed.h"
#include "uTensor_util.hpp"

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
                accm += std::numeric_limits<float>::quiet_NaN();
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

#endif
