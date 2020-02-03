#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include "operatorBase.hpp"

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
       c(i) = static_cast<T>(a(i)) + static_cast<T>(b(i));
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

}
#endif
