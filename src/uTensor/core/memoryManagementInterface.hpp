#ifndef __UTENSOR_MEMORY_MANAGEMENT_IFC_H
#define __UTENSOR_MEMORY_MANAGEMENT_IFC_H
#include <cstring>
namespace uTensor {

class AllocatorInterface;

class Handle {
 private:
  Handle(const Handle& that);
  // Handles cannot be allocated directly
  void* operator new(size_t sz);
  void operator delete(void* p);
  // KEY BIT
  friend class AllocatorInterface;

 public:
  Handle();
  Handle(void* p);
  // return the data directly (looks pointer like)
  void* operator*();

 protected:
  void* _ptr;
};

class RawDataHandle : public Handle {
 public:
  RawDataHandle(size_t req_ram_size) {
    _ptr = Context::DefaultRamDataAllocator::allocate(req_ram_size);
    Context::DefaultRamDataAllocator::bind(this, _ptr);
  }
  ~RawDataHandle() { Context::DefaultRamDataAllocator::deallocate(_ptr); }
  // void* operator new(size_t sz) { // Have to delegate this size from tensors
  // somehow + sizeof(Tensor)
  //  void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz);
  //  return p;
  //}
  // void operator delete(void* p) {
  //  Context::DefaultTensorMetaDataAllocator::deallocate(p);
  //}
  // Ignore this, use the constructor to specify the data space
  //  void* operator new(size_t sz, size_t req_ram_size){
  //    void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz);
  //    _ptr = Context::DefaultRamDataAllocator::allocator(req_ram_size);
  //    return p;
  //  }
};
/**
 * Allocators are expected to maintain a mapping of Tensor handles to data
 * regions. * This allows the allocator to move around the underlying data
 * without breaking the user interface.
 */
class AllocatorInterface {
  // Allocators must implement these functions
 protected:
  virtual void _bind(void* ptr, Handle* hndl) = 0;
  virtual void _unbind(void* ptr, Handle* hndl) = 0;
  virtual bool _is_bound(void* ptr, Handle* hndl) = 0;
  virtual bool _has_handle(Handle* hndl) = 0;

 public:
  /*
   * Public interface for updating a Tensor Handle reference
   */
  void update_hndl(Handle& h, void* new_ptr);

  /**
   * Bind/Unbind data to Tensor Handle
   */
  void bind(void* ptr, Handle* hndl);
  void unbind(void* ptr, Handle* hndl);
  /**
   * Check if a pointer is associated with a Tensor
   */
  bool is_bound(void* ptr, Handle* hndl);

  /**
   * Returns the amount of space available in the Memory Manager
   */
  virtual size_t available() = 0;

  /**
   * Update Tensor handles to point to new regions.
   * This is useful is the data moves around inside the memory manager,
   * For example if the data is compressed/decompressed dynamically
   */
  virtual bool
  rebalance() = 0;  // KEY. This call updates all the Tensor data references

  /**
   * Allocate sz bytes in the memory manager
   */
  virtual void* allocate(size_t sz) = 0;

  /**
   * Deallocate all data associated with pointer
   */
  virtual void deallocate(void* ptr) = 0;
};

}  // namespace uTensor
#endif
