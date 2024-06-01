#include "allocator.hpp"

#include <cstdlib>
#include <iostream>

using std::free;
using std::malloc;

static uTensor::MallocAllocator ram_allocator(0);
static uTensor::MallocAllocator meta_allocator(0);

namespace uTensor {

DEFINE_ERROR(InvalidBoundError);

MallocAllocator::MallocAllocator(size_t capability)
    : total_(capability), accum_(0) {}
size_t MallocAllocator::available() {
  size_t rem = static_cast<size_t>(total_ - accum_);
  return rem;
}
bool MallocAllocator::rebalance() { return true; }
void MallocAllocator::set_total(size_t v) { total_ = v; }

void MallocAllocator::_bind(void *ptr, Handle *hndl) {
  if (ptrs_map_.find(hndl) != ptrs_map_.end()) {
    Context::get_default_context()->throwError(new InvalidBoundError());
  }
  ptrs_map_[hndl] = ptr;
}
void MallocAllocator::_unbind(void *ptr, Handle *hndl) {
  if (ptrs_map_.find(hndl) == ptrs_map_.end()) {
    return;
  }
  ptrs_map_.erase(hndl);
}
bool MallocAllocator::_is_bound(void *ptr, Handle *hndl) {
  if (ptrs_map_.find(hndl) == ptrs_map_.end()) {
    return false;
  }
  return ptrs_map_[hndl] == ptr;
}
bool MallocAllocator::_has_handle(Handle *hndl) {
  return ptrs_map_.find(hndl) != ptrs_map_.end();
}
void *MallocAllocator::_allocate(size_t sz) {
  if (total_ < accum_ + sz) {
    std::cerr << "need to enlarge the total memory usage" << std::endl;
    throw std::bad_alloc();
  }
  void *addr = malloc(sz);
  accum_ += sz;
  size_map_[addr] = sz;
  return addr;
}
void MallocAllocator::_deallocate(void *ptr) {
  if (size_map_.find(ptr) == size_map_.end()) {
    // throw error?
    return;
  }
  size_t sz = size_map_[ptr];
  size_map_.erase(ptr);
  accum_ -= sz;
  free(ptr);
}

MallocAllocator *python::get_ram_allocator() { return &ram_allocator; }
MallocAllocator *python::get_meta_allocator() { return &meta_allocator; }
void python::set_ram_total(size_t capacity) {
  ram_allocator.set_total(capacity);
}
void python::set_meta_total(size_t capacity) {
  meta_allocator.set_total(capacity);
}

}  // namespace uTensor
