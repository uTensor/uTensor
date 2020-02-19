#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H
#include "errorHandler.hpp"
#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"

namespace uTensor {

DECLARE_ERROR(MemoryAllocatorUnsetError);

class Context {
 private:
  Context();

 public:
  static Context* get_default_context();
  ErrorHandler* get_default_error_handler();
  AllocatorInterface* get_metadata_allocator();
  AllocatorInterface* get_ram_data_allocator();
  void set_metadata_allocator(AllocatorInterface* al);
  void set_ram_data_allocator(AllocatorInterface* al);
  void set_ErrorHandler(ErrorHandler* errH);
  void register_tensor(TensorBase* tb);
  void throwError(Error* err);
  void notifyEvent(const Event& err);

 private:
  AllocatorInterface* _metadata_allocator;
  AllocatorInterface* _ram_data_allocator;
  ErrorHandler* _error_handler;
};
}  // namespace uTensor
#endif  // UTENSOR_CTX_H
