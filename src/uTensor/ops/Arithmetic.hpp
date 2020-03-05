#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include "operatorBase.hpp"
#include <algorithm>
#include <limits>

namespace uTensor {

template<typename T>
void add_kernel(Tensor& c, const Tensor& a, const Tensor& b){
    // Decide on c shape
    TensorShape c_shape = c->get_shape();
    uint32_t c_size = c_shape.get_linear_size();
    //TensorInterface& C = reinterpret_cast<TensorInterface*>(*c);
    //const TensorInterface& A = reinterpret_cast<TensorInterface*>(*a);
    //const TensorInterface& B = reinterpret_cast<TensorInterface*>(*b);


    for (uint32_t i = 0; i < c_size; i++)
       c(i) = static_cast<T>(static_cast<T>(a(i)) + static_cast<T>(b(i)));
}

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
    enum names_in: uint8_t {a};
    enum names_out: uint8_t {out};
    //AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) : OperatorBase(inputs, outputs) {}

protected:
    virtual void compute() {
      T tmp = std::numeric_limits<T>::max();
      const Tensor& a_t = *inputs[a].tensor;
      Tensor& out_t = *outputs[out].tensor;
      const TensorShape& a_shape = a_t->get_shape();
      const uint32_t a_size = a_shape.get_linear_size();
      for(uint32_t i = 0; i < a_size; i++){
        tmp = std::min(tmp, static_cast<T>(a_t(i)));
      }
      out_t(0) = tmp;
    }
};

template<typename T>
class MaxOperator : public OperatorInterface<1, 1> {
public:
    enum names_in: uint8_t {a};
    enum names_out: uint8_t {out};
    //AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) : OperatorBase(inputs, outputs) {}

protected:
    virtual void compute() {
      T tmp = std::numeric_limits<T>::lowest();
      const Tensor& a_t = *inputs[a].tensor;
      Tensor& out_t = *outputs[out].tensor;
      const TensorShape& a_shape = a_t->get_shape();
      const uint32_t a_size = a_shape.get_linear_size();
      for(uint32_t i = 0; i < a_size; i++){
        tmp = std::max(tmp, static_cast<T>(a_t(i)));
      }
      out_t(0) = tmp;
    }
};



}
#endif
