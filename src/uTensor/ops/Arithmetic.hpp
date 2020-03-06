#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include <algorithm>
#include <limits>
#include "operatorBase.hpp"
#include "Arithmetic_kernels.hpp"
#include "functional_kernels.hpp"

namespace uTensor {

template<typename T>
class AddOperator : public OperatorInterface<2, 1> {
public:
    enum names_in: uint8_t {a, b};
    enum names_out: uint8_t {c};
    //AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) : OperatorBase(inputs, outputs) {}

protected:
    virtual void compute() {
        add_kernel<T>(*outputs[c].tensor, *inputs[a].tensor, *inputs[b].tensor);
    }
};


template<typename T>
class MinOperator : public OperatorInterface<1, 1> {
public:
    enum names_in: uint8_t {in};
    enum names_out: uint8_t {out};

protected:
    virtual void compute() {
      min_kernel<T>(*outputs[out].tensor, *inputs[in].tensor);
    }
};

template<typename T>
class MaxOperator : public OperatorInterface<1, 1> {
public:
    enum names_in: uint8_t {in};
    enum names_out: uint8_t {out};

protected:
    virtual void compute() {
      max_kernel<T>(*outputs[out].tensor, *inputs[in].tensor);
    }
};



}
#endif
