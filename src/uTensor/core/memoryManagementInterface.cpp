#include "memoryManagementInterface.hpp"

#include "uTensor_util.hpp"

namespace uTensor {

Handle::Handle(const Handle& that) {}
void* Handle::operator new(size_t sz) { return nullptr; }
void Handle::operator delete(void* p) {}
Handle::Handle() : _ptr(nullptr) {}
Handle::Handle(void* p) : _ptr(p) {}
void* Handle::operator*() { return _ptr; }
bool Handle::operator!() const { return _ptr == nullptr; }

void AllocatorInterface::update_hndl(Handle* h, void* new_ptr) {
  h->_ptr = new_ptr;
}

void AllocatorInterface::bind(void* ptr, Handle* hndl) {
  if (!_has_handle(hndl))
    DEBUG("Allocator does not contain reference to handle");

  if (is_bound(ptr, hndl)) {
    ERR_EXIT("Cannot rebind Handles without unbinding");
  }
  _bind(ptr, hndl);
}
void AllocatorInterface::unbind(void* ptr, Handle* hndl) {
  if (!is_bound(ptr, hndl)) {
    ERR_EXIT("Cannot unbind unbound Handles");
  }
  _unbind(ptr, hndl);
}
bool AllocatorInterface::is_bound(void* ptr, Handle* hndl) {
  return _is_bound(ptr, hndl);
}
void* AllocatorInterface::allocate(size_t sz) {
    if(sz > (1 << 16)){
        //TODO ERROR invalid allocation size
        return nullptr;
    }
    return _allocate(sz);
}

void AllocatorInterface::deallocate(void* ptr) {
    _deallocate(ptr);
}

};  // namespace uTensor
