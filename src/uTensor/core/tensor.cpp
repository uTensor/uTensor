#include "tensor.hpp"

namespace uTensor {
Tensor::Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by reference

TensorInterface* Tensor::operator->(0) { return _ptr; }
Tensor::Tensor(utensor::TensorInterface* ptr) : _ptr(ptr) {
  Context::DefaultTensorMetaDataAllocator::bind(this, ptr);
}
// Add some bits to make the interface nicer to the user

// Force everything to be on the utensor allocator
void* Tensor::operator new(size_t sz) { // Have to delegate this size from tensors somehow + sizeof(Tensor)
  void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz); 
  return p;
}
void Tensor::operator delete(void* p) {
  Context::DefaultTensorMetaDataAllocator::deallocate(p);
}

SimpleNamedTensor::SimpleNamedTensor(const uTensor::string& name, Tensor& tensor) : name(name), tensor(tensor) {}
}
