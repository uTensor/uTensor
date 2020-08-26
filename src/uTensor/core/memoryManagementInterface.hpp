#ifndef __UTENSOR_MEMORY_MANAGEMENT_IFC_H
#define __UTENSOR_MEMORY_MANAGEMENT_IFC_H

// Don't see where cstring is used here
//#include <cstring>

// #include <cstddef>
// using std::size_t;

#include "stdint.h" //avr
using namespace std;

namespace uTensor {

class AllocatorInterface;

// TODO Add support for Moving Handles without copies, Must be careful with
// binding
class Handle {
 private:
  // Handles cannot be copied, there exists one unless explicitly deep copied
  // TODO write Handle deep_copy(const Handle& that);
  Handle(const Handle& that);
  // Handles cannot be allocated directly
  void* operator new(size_t sz);
  void operator delete(void* p);
  // KEY BIT
  friend class AllocatorInterface;
  friend void* operator*(const Handle& that);

 public:
  Handle();
  Handle(void* p);
  // return the data directly (looks pointer like)
  void* operator*();
  // Allow users to check if handle is not valid
  bool operator!() const;
  operator bool() const;

 protected:
  void* _ptr;
};

void* operator*(const Handle& that);

/**
 * Expecting the Handle copies to contain knowledge of their underlying
 * types/copy info is not reasonable. This class is designed to reference a
 * singleton-like handle for some data
 */
class HandleReference {
 protected:
  Handle* _ref;

 public:
  HandleReference();
  HandleReference(Handle* ref);
  HandleReference(const Handle& ref);
  HandleReference(const HandleReference& that);
  HandleReference(HandleReference&& that);
  HandleReference& operator=(Handle* ref);
  HandleReference& operator=(const HandleReference& that);
  HandleReference& operator=(HandleReference&& that);

  // Delegate functions
  // return the data directly (looks pointer like)
  void* operator*();
  // Allow users to check if handle is not valid
  bool operator!() const;
  operator bool() const;
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
  virtual void* _allocate(size_t sz) = 0;
  virtual void _deallocate(void* ptr) = 0;

 public:
  /*
   * Public interface for updating a Tensor Handle reference
   */
  void update_hndl(Handle* h, void* new_ptr);

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
  void* allocate(size_t sz);

  /**
   * Deallocate all data associated with pointer
   */
  void deallocate(void* ptr);

  /** Unbind invalidates a handle, so rather than forcing users to store a temporary, 
   * this conveniece function does it for them,
   */
  void unbind_and_deallocate(Handle* hndl);
};

bool bind(Handle& hndl, AllocatorInterface& allocator);
bool unbind(Handle& hndl, AllocatorInterface& allocator);
bool is_bound(Handle& hndl, AllocatorInterface& allocator);
}  // namespace uTensor
#endif
