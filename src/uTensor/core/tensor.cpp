#include "tensor.hpp"
#include "context.hpp"

namespace uTensor {
// Tensor::Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by
// reference

TensorInterface* Tensor::operator->() {
  return reinterpret_cast<TensorInterface*>(_ptr);
}
Tensor::Tensor(TensorInterface* ptr) : Handle((void*)ptr) {
  // Context::get_metadata_allocator()->bind(ptr, this);
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

SimpleNamedTensor::SimpleNamedTensor(const uTensor::string& name,
                                     Tensor& tensor)
    : name(name), tensor(tensor) {}
}  // namespace uTensor
