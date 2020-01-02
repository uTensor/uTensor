#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H
#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"

namespace uTensor {
class Context {
  // private:
  //  Context();

 public:
  static Context* get_default_context();
  static AllocatorInterface* get_metadata_allocator();
  static AllocatorInterface* get_ram_data_allocator();
  static void set_metadata_allocator(AllocatorInterface* al);
  static void set_ram_data_allocator(AllocatorInterface* al);
  void register_tensor(TensorBase* tb);

 private:
  static AllocatorInterface* _metadata_allocator;
  static AllocatorInterface* _ram_data_allocator;
};
}  // namespace uTensor
#endif  // UTENSOR_CTX_H
