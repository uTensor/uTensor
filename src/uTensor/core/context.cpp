#include "uTensor/core/context.hpp"
#include "uTensor/core/uTensor_util.hpp"
namespace uTensor {

// AllocatorInterface* Context::_metadata_allocator = nullptr;
// AllocatorInterface* Context::_ram_data_allocator = nullptr;

// EVENT LIST
DEFINE_ERROR(MemoryAllocatorUnsetError);

Context::Context()
    : _metadata_allocator(nullptr),
      _ram_data_allocator(nullptr),
      _error_handler(nullptr) {}

Context* __attribute__((weak)) Context::get_default_context() {
  static Context ctx;
  return &ctx;
}
AllocatorInterface* Context::get_metadata_allocator() {
  // TODO Throw error if this is null
  if (!_metadata_allocator) {
    uTensor_printf("[ERROR] Metadata allocator not set in context before it was read\n");
    throwError(new MemoryAllocatorUnsetError);
  }
  return _metadata_allocator;
}
AllocatorInterface* Context::get_ram_data_allocator() {
  // TODO Throw error if this is null
  if (!_ram_data_allocator) {
    uTensor_printf("[ERROR] Ramdata allocator not set in context before it was read\n");
    throwError(new MemoryAllocatorUnsetError);
  }
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
  if (_error_handler){
    get_default_error_handler()->uThrow(err);
    return;
  }
  uTensor_printf("[ERROR] an Error has occurred but no handler was set\n");
  //while(true) {}
}
void Context::notifyEvent(const Event& evt) {
  if (_error_handler) get_default_error_handler()->notify(evt);
}
void Context::set_ErrorHandler(ErrorHandler* errH) { _error_handler = errH; }

// ErrorHandler default_err;
ErrorHandler* Context::get_default_error_handler() {
  // static ErrorHandler err{};
  if (!_error_handler) {
    //_error_handler = &default_err;
  }
  return _error_handler;
}

}  // namespace uTensor
