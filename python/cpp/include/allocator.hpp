#ifndef UTENSOR_ALLOCATOR_H
#define UTENSOR_ALLOCATOR_H

#include <unordered_map>

#include "uTensor.h"
#include "uTensor/core/errorHandler.hpp"

namespace uTensor {

DECLARE_ERROR(InvalidBoundError);

class MallocAllocator : public AllocatorInterface {
 public:
  MallocAllocator(size_t capability);
  size_t available() override;
  bool rebalance() override;
  void set_total(size_t v);

 protected:
  void _bind(void *ptr, Handle *hndl) override;
  void _unbind(void *ptr, Handle *hndl) override;
  bool _is_bound(void *ptr, Handle *hndl) override;
  bool _has_handle(Handle *hndl) override;
  void *_allocate(size_t sz) override;
  void _deallocate(void *ptr) override;

 private:
  std::unordered_map<Handle *, void *> ptrs_map_;
  std::unordered_map<void *, size_t> size_map_;
  unsigned long long total_;
  unsigned long long accum_;
};

namespace python {
MallocAllocator *get_ram_allocator();
MallocAllocator *get_meta_allocator();

void set_ram_total(size_t capacity);
void set_meta_total(size_t capacity);
}  // namespace python

}  // namespace uTensor

#endif