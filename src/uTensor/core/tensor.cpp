#include "tensor.hpp"
#include "context.hpp"

namespace uTensor {
// Tensor::Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by
// reference

TensorInterface* Tensor::operator->() {
  return reinterpret_cast<TensorInterface*>(_ptr);
}
TensorInterface* Tensor::operator*() {
  return reinterpret_cast<TensorInterface*>(_ptr);
}
Tensor::Tensor(TensorInterface* ptr) : Handle((void*)ptr) {
  // Context::get_metadata_allocator()->bind(ptr, this);
}
Tensor& Tensor::operator=(TensorInterface* ptr) {
  _ptr = (void*)ptr;
  return *this;
}

Tensor::Tensor(Tensor&& that) {
    _ptr = that._ptr;
    that._ptr = nullptr;
}
Tensor& Tensor::operator=(Tensor&& that) {
  if (this != &that) {
    _ptr = that._ptr;
    that._ptr = nullptr;
  }
  return *this;
}
// Add some bits to make the interface nicer to the user

// Force everything to be on the utensor allocator
void* Tensor::operator new(size_t sz) {  // Have to delegate this size from
                                         // tensors somehow + sizeof(Tensor)
  void* p = Context::get_metadata_allocator()->allocate(sz);
  return p;
}
void Tensor::operator delete(void* p) {
  Context::get_metadata_allocator()->deallocate(p);
}

// Interface
const IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                               uint16_t l) const {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k, l);
}
IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                         uint16_t l){
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k, l);
}
const IntegralValue Tensor::operator()(uint32_t linear_index) const {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(linear_index);
}
IntegralValue Tensor::operator()(uint32_t linear_index){
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(linear_index);

}


TensorInterface* TensorReference::operator*(){
  return reinterpret_cast<TensorInterface*>(_ref->operator*()); 
}

SimpleNamedTensor::SimpleNamedTensor(const uTensor::string& name,
                                     Tensor& tensor)
    : name(&name), tensor(&tensor) {}
SimpleNamedTensor::SimpleNamedTensor() {}



}  // namespace uTensor

