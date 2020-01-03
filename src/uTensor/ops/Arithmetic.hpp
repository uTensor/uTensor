#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
namespace uTensor {

template<typename T>
void add_kernel(Tensor& c, const Tensor& a, const Tensor& b){
    // Decide on c shape
    for (int i = 0; i < c.size(); i++)
        c[i] = static_cast<T>(a[i]) + static_cast<T>(b[i]);
}

template<typename T>
class AddOperator : public OperatorInterface<num_inputs=2, num_outputs=1> {
public:
    static enum names: uint8_t {a, b, c};
    //AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) : OperatorBase(inputs, outputs) {}

protected:
    virtual void compute() {
        add_kernel<T>(outputs[c], inputs[a], inputs[b]);
    }

};

}
#endif
