#include "context.hpp"
namespace uTensor {

AllocatorInterface* Context::_metadata_allocator = nullptr;
AllocatorInterface* Context::_ram_data_allocator = nullptr;

Context* Context::get_default_context() {
  static Context ctx;
  return &ctx;
}
AllocatorInterface* Context::get_metadata_allocator() {
  // TODO Throw error if this is null
  return _metadata_allocator;
}
AllocatorInterface* Context::get_ram_data_allocator() {
  // TODO Throw error if this is null
  return _ram_data_allocator;
}
void Context::set_metadata_allocator(AllocatorInterface* al) {
  // TODO handle cleanup if already set?
  _metadata_allocator = al;
}
void Context::set_ram_data_allocator(AllocatorInterface* al) {
  // TODO handle cleanup if already set?
  _ram_data_allocator = al;
}
void Context::register_tensor(TensorBase* tb) {}

void Context::throwError(Error* err) {
    _error_handler.uThrow(err);
}

}  // namespace uTensor

