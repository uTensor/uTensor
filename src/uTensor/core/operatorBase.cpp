#include "operatorBase.hpp"
#include "context.hpp"

namespace uTensor {

//EVENTS AND ERRORS
DEFINE_ERROR(OperatorIOSizeMismatchError);

// OperatorBase::OperatorBase(TensorMapInterface* inputs, TensorMapInterface*
// outputs) : inputs(inputs), outputs(outputs) {
//    Context* ctx = Context::get_default_context();
//    //        ctx.push_op_tensors(*this, inputs);
//    //        ctx.push_op_tensors(*this, outputs);
//}
// The preferred interface
OperatorBase::OperatorBase(TensorMapInterface* inputs) : _p_inputs(inputs) {
  Context* ctx = Context::get_default_context();
  // ctx.push_op_tensors(*this, inputs);
}
OperatorBase::OperatorBase() {}
OperatorBase::OperatorBase(uTensor::string op_name) : op_name(op_name) {}

void OperatorBase::set_inputs(TensorMapInterface* inputs) {
  this->_p_inputs = inputs;
  Context* ctx = Context::get_default_context();
  // ctx.push_op_tensors(*this, inputs);
}
void OperatorBase::set_outputs(TensorMapInterface* outputs) {
  this->_p_outputs = outputs;
  Context* ctx = Context::get_default_context();
  // ctx.push_op_tensors(*this, outputs);
}

void OperatorBase::eval() { compute(); }
OperatorBase::~OperatorBase() {
  Context* ctx = Context::get_default_context();
  // ctx.pop_op_tensors(*this, inputs); // Inputs are no longer needed
}
void OperatorBase::set_name(uTensor::string _name) { op_name = _name; }

size_t FastOperator::get_readable_block(const Tensor& t, void* buffer,
                                        uint16_t num_elems, int linear_index) {
  return t->get_readable_block(buffer, num_elems, linear_index);
}
size_t FastOperator::get_writeable_block(Tensor& t, void* buffer,
                                         uint16_t num_elems, int linear_index) {
  return t->get_writeable_block(buffer, num_elems, linear_index);
}

}  // namespace uTensor
