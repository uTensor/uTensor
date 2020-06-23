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
Handle::operator bool() const { return _ptr != nullptr; }

void* operator*(const Handle& that) { return that._ptr; }

/** Handle Reference Stuff
 */
HandleReference::HandleReference() : _ref(nullptr) {}
HandleReference::HandleReference(Handle* ref) : _ref(ref) {}
HandleReference::HandleReference(const Handle& ref)
    : _ref(const_cast<Handle*>(&ref)) {}
HandleReference::HandleReference(const HandleReference& that) {
  _ref = that._ref;
}
HandleReference::HandleReference(HandleReference&& that) {
  _ref = that._ref;
  that._ref = nullptr;
}
HandleReference& HandleReference::operator=(Handle* ref) {
  _ref = ref;
  return *this;
}
HandleReference& HandleReference::operator=(const HandleReference& that) {
  _ref = that._ref;
  return *this;
}
HandleReference& HandleReference::operator=(HandleReference&& that) {
  if (this != &that) {
    _ref = that._ref;
    that._ref = nullptr;
  }
  return *this;
}

// Delegate functions
// return the data directly (looks pointer like)
void* HandleReference::operator*() {
  if (_ref) {
    // Hue hue pointer like
    return **_ref;
  }
  return nullptr;
}
// Allow users to check if handle is not valid
bool HandleReference::operator!() const {
  if (_ref) {
    return !((bool)_ref);
  }
  return true;
}
HandleReference::operator bool() const {
  if (_ref) {
    return ((bool)_ref);
  }
  return true;
}

void AllocatorInterface::update_hndl(Handle* h, void* new_ptr) {
  h->_ptr = new_ptr;
}

void AllocatorInterface::bind(void* ptr, Handle* hndl) {
  if (!_has_handle(hndl))
    DEBUG("Allocator does not contain reference to handle");

  if (is_bound(ptr, hndl)) {
    ERR_EXIT("Cannot rebind Handles without unbinding");
  }
  // To be safe, make sure handle has a pointer
  update_hndl(hndl, ptr);
  _bind(ptr, hndl);
}
void AllocatorInterface::unbind(void* ptr, Handle* hndl) {
  if (!is_bound(ptr, hndl)) {
    ERR_EXIT("Cannot unbind unbound Handles");
  }
  _unbind(ptr, hndl);
  update_hndl(hndl, nullptr);
}
bool AllocatorInterface::is_bound(void* ptr, Handle* hndl) {
  return _is_bound(ptr, hndl);
}
void* AllocatorInterface::allocate(size_t sz) {
  if (sz > (1 << 31)) {
    // TODO ERROR invalid allocation size
    uTensor_printf("[ERROR] Attempted to allocator > 2**32 bytes\n");
    return nullptr;
  }
  return _allocate(sz);
}

void AllocatorInterface::deallocate(void* ptr) {
  _deallocate(ptr);
  ptr = nullptr;
}

void AllocatorInterface::unbind_and_deallocate(Handle* hndl){
  void* ptr_t = hndl->_ptr;
  if (hndl->_ptr) {
    if (this->is_bound(hndl->_ptr, hndl)) {
      this->unbind(hndl->_ptr, hndl);
    } else {
      ERR_EXIT("Cannot unbind unbound Handles");
    }
    this->deallocate(ptr_t);
  }

}

bool bind(Handle& hndl, AllocatorInterface& allocator) {
  if (!hndl) {
    return false;
  }
  allocator.bind(*hndl, &hndl);
  return true;
}
bool unbind(Handle& hndl, AllocatorInterface& allocator) {
  if (!hndl) {
    return false;
  }
  allocator.unbind(*hndl, &hndl);
  return true;
}
bool is_bound(Handle& hndl, AllocatorInterface& allocator) {
  return allocator.is_bound(*hndl, &hndl);
}

};  // namespace uTensor
