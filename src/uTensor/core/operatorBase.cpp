#include "operatorBase.hpp"
#include "context.hpp"

namespace uTensor {

//OperatorBase::OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs) : inputs(inputs), outputs(outputs) {
//    Context* ctx = Context::get_default_context();
//    //        ctx.push_op_tensors(*this, inputs);
//    //        ctx.push_op_tensors(*this, outputs);
//}
  // The preferred interface
OperatorBase::OperatorBase(TensorMapInterface* inputs) : _p_inputs(inputs) {
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, inputs);
}
OperatorBase::OperatorBase() {}
OperatorBase::OperatorBase(uTensor::string op_name) : op_name(op_name) {}

void OperatorBase::set_inputs(TensorMapInterface* inputs) {
    this->_p_inputs = inputs; 
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, inputs);
}
void OperatorBase::set_outputs(TensorMapInterface* outputs) {
    this->_p_outputs = outputs; 
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, outputs);
}

void OperatorBase::eval() {
  compute();
}
OperatorBase::~OperatorBase() {
    Context* ctx = Context::get_default_context();
    //ctx.pop_op_tensors(*this, inputs); // Inputs are no longer needed
}
void OperatorBase::set_name(uTensor::string _name) {
  op_name = _name;
}
}
