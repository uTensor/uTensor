#ifndef UTENSOR_ARENA_ALLOCATOR_HPP
#define UTENSOR_ARENA_ALLOCATOR_HPP
#include <algorithm>  // for sort function
#include <cstdio>
//#include <forward_list>
#include <memory>
#include <vector>

#include "uTensor/core/context.hpp"
#include "uTensor/core/memoryManagementInterface.hpp"
#include "uTensor/core/tensor.hpp"

namespace uTensor {

//#define MSB_SET ~( ~( (T)0 ) >> 1 )
#define MSB_SET (1 << (sizeof(uint32_t) * 8 - 1))
#define BLOCK_INACTIVE ~MSB_SET
#define BLOCK_LENGTH_MASK ~MSB_SET
#define BLOCK_ACTIVE MSB_SET
#define BLOCK_ZERO_LENGTH 0

/** EVENT LIST
 */

DECLARE_EVENT(MetaHeaderNotFound);
DECLARE_EVENT(localCircularArenaAllocatorRebalancing);
DECLARE_EVENT(localCircularArenaAllocatorConstructed);
DECLARE_ERROR(InvalidBoundRegionState);
DECLARE_ERROR(MetaHeaderNotBound);
DECLARE_ERROR(InvalidAlignmentAllocation);

template <typename T>
constexpr size_t meta_addressable_space() {
  return 1 << (sizeof(T) * 8 - 1);
}
/**
 * Size allocated must be less than 2**15
 * TODO get around the BS alignment bits from the silly pointer variable causing
 * extra empty padding
 * Break this up into two classes to not duplicate code
 */
class localCircularArenaAllocatorBase : public AllocatorInterface {
 private:

  class MetaHeader {
   public:
    uint32_t meta_data;
    Handle* hndl;
    uint8_t* _d;

   public:
    MetaHeader();
    MetaHeader(uint32_t sz);
    MetaHeader(uint32_t sz, uint8_t* d);
    void set_active();
    void set_inactive();
    void set_hndl(Handle* handle); 
    void set_d(uint8_t* d);
    void set_len(uint32_t sz);
    uint32_t get_len() const ;
    bool is_active() const ;
    bool is_bound() const ;
    bool has_handle(Handle* target) const ;
    bool is_used() const ;
  };

 protected:
  uint32_t capacity;
  uint8_t* _buffer;
  size_t size;
  uint8_t* cursor;
  std::vector<MetaHeader> _headers;

 private:
  // Return the amount of free space at the tail
  uint32_t tail_capacity();

  size_t find_header_associated_w_ptr(void* ptr) const;
  // This is just for reference
  MetaHeader& _read_header(void* ptr);

  /*
  void _write_header(const MetaHeader& hdr, void* ptr) {
    // First check if ptr in bounds
    if (ptr < _buffer || ptr > (_buffer + size)) {
      // ERROR
    }
    _headers.push_back(hdr);
  }
  */

  void* attempt_to_reuse_inactive_region(size_t sz); 
  
  inline uint8_t* begin() const; 
  inline const uint8_t* end() const; 
  inline size_t _get_size() const; 

 protected:
  virtual void _bind(void* ptr, Handle* hndl); 

  virtual void _unbind(void* ptr, Handle* hndl); 

  virtual bool _is_bound(void* ptr, Handle* hndl); 

  virtual bool _has_handle(Handle* hndl); 

  virtual void* _allocate(size_t sz);

  virtual void _deallocate(void* ptr);

 public:
  localCircularArenaAllocatorBase(uint8_t* buffer, size_t size) ;
  virtual ~localCircularArenaAllocatorBase() ;

  /** This implementation of rebalance shifts all allocated chunks to the end of
   * the buffer and inserts an inactive region at the start. note: cursor gets
   * moved to begin() note: unbound regions get wiped
   */
  // TODO Check to make sure updated locations are still aligned
  virtual bool rebalance();

  virtual size_t available(); 

  virtual void clear(); 

  // Check to see if pointer exists in memory space and is valid
  bool contains(void* p) const; 

 public:
  // Testing bits, attribute out later
  uint32_t internal_header_unit_size() const; 
};

template <size_t sizeT, typename T = uint16_t>
class localCircularArenaAllocator : public localCircularArenaAllocatorBase {
  private:
  static constexpr size_t size = sizeT;
  // Compile time check to make sure that the user is using a T large enough to
  // address size
  static_assert(size < meta_addressable_space<T>(),
                "[ERROR](localCircularArenaAllocator) T not large enoughn to "
                "address size in Arena. Attempted to create Arena with (size, "
                "T) mismatch, try increasing T to uint32_t");
  private:
    // Delegate this over to the implementation
    uint8_t _buffer[size];

  public:
    localCircularArenaAllocator() : localCircularArenaAllocatorBase(_buffer, size), _buffer{} {
      //memset(_buffer, 0, size);
    }
    virtual ~localCircularArenaAllocator() {}

};

// Note not actually complete
template <size_t sz>
class ArenaCircularAllocator {
 private:
  static localCircularArenaAllocator<sz> _allocator;

 public:
  static void* allocate(size_t size) {
    void* p = _allocator.allocate(size);
    return p;
  }
  static void deallocate(void* p) { _allocator.deallocate(p); }
  static void bind(void* ptr, Tensor* hndl) { _allocator.bind(ptr, hndl); }
  static void unbind(void* ptr, Tensor* hndl) { _allocator.unbind(ptr, hndl); }
};

}  // namespace uTensor
#endif
