#include "operatorBase.hpp"
#include "context.hpp"

namespace uTensor {

OperatorBase::OperatorBase(TensorMapInterface* inputs, TensorMapInterface* outputs) : inputs(inputs), outputs(outputs) {
    Context* ctx = Context::get_default_context();
    //        ctx.push_op_tensors(*this, inputs);
    //        ctx.push_op_tensors(*this, outputs);
}
  // The preferred interface
OperatorBase::OperatorBase(TensorMapInterface* inputs) : inputs(inputs) {
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, inputs);
}
OperatorBase::OperatorBase() {}

void OperatorBase::set_inputs(TensorMapInterface* inputs) {
    this->inputs = inputs; 
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, inputs);
}
void OperatorBase::set_outputs(TensorMapInterface* outputs) {
    this->outputs = outputs; 
    Context* ctx = Context::get_default_context();
    //ctx.push_op_tensors(*this, outputs);
}
OperatorBase::~OperatorBase() {
    Context* ctx = Context::get_default_context();
    //ctx.pop_op_tensors(*this, inputs); // Inputs are no longer needed
}
void OperatorBase::set_name(uTensor::string _name) {
  op_name = _name;
}
}
