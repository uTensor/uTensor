#ifndef UTENSOR_ARENA_ALLOCATOR_HPP
#define UTENSOR_ARENA_ALLOCATOR_HPP
#include <cstdio>
#include <forward_list>
#include <vector>
#include <algorithm> // for sort function
#include <memory>
#include "memoryManagementInterface.hpp"
#include "tensor.hpp"

namespace uTensor {

//#define MSB_SET ~( ~( (T)0 ) >> 1 )
#define MSB_SET (1 << (sizeof(T) * 8 - 1))
#define BLOCK_INACTIVE ~MSB_SET
#define BLOCK_LENGTH_MASK ~MSB_SET
#define BLOCK_ACTIVE MSB_SET
#define BLOCK_ZERO_LENGTH 0

/**
 * Size allocated must be less than 2**15
 * TODO get around the BS alignment bits from the silly pointer variable causing
 * extra empty padding
 */
template <size_t size, typename T = uint16_t>
class localCircularArenaAllocator : public AllocatorInterface {
 private:
  class MetaHeader {
   public:
    T meta_data;
    Handle* hndl;
    uint8_t* _d;

   public:
    MetaHeader()
        : meta_data(BLOCK_INACTIVE & BLOCK_ZERO_LENGTH), hndl(nullptr), _d(nullptr) {}
    MetaHeader(T sz) : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(nullptr) {}
    MetaHeader(T sz, uint8_t* d) : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr), _d(d) {}
    void set_active() { meta_data |= BLOCK_ACTIVE; }
    void set_inactive() { meta_data &= BLOCK_INACTIVE; }
    void set_hndl(Handle* handle) { hndl = handle; }
    void set_d(uint8_t* d) { _d = d; }
    void set_len(T sz) {
      meta_data &= MSB_SET;  // Clear all size bits
      meta_data |= (BLOCK_LENGTH_MASK & sz);
    }
    T get_len() const { return meta_data & BLOCK_LENGTH_MASK; }
    bool is_active() const { return (meta_data & MSB_SET) == BLOCK_ACTIVE; }
    bool is_bound() const { return (hndl != nullptr); }
    bool has_handle(Handle* target) const {
      return is_active() && (hndl == target);
    }
    bool is_used() const { return is_active() && (get_len() > 0); }
  };


 private:
  T capacity;
  uint8_t _buffer[size];
  uint8_t* cursor;
  std::vector<MetaHeader> _headers;;

 private:
  // Return the amount of free space at the tail
  T tail_capacity(){};

  size_t find_header_associated_w_ptr(void* ptr) const {
    size_t i = 0;
    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
      if(hdr_i->_d == ptr)
        return i;
    }
    return i;
  }
  // This is just for reference
  MetaHeader& _read_header(void* ptr) {
    static MetaHeader not_found;
    // First check if ptr in bounds
    if (ptr < _buffer || ptr > (_buffer + size)) {
      // ERROR
    }
    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
      if(hdr_i->_d == ptr)
        return *hdr_i;
    }
    // ERROR
    return not_found;
  }

  uint8_t* begin() {
    return _buffer;
  }

  uint8_t* end() {
    return _buffer + size;
  }
  /*
  void _write_header(const MetaHeader& hdr, void* ptr) {
    // First check if ptr in bounds
    if (ptr < _buffer || ptr > (_buffer + size)) {
      // ERROR
    }
    _headers.push_back(hdr);
  }
  */

 protected:
  virtual void _bind(void* ptr, Handle* hndl) {
    MetaHeader& hdr = _read_header(ptr);
    // Check if region is active
    if (!hdr.is_active()) {
      // ERROR
    }
    hdr.set_hndl(hndl);
  }

  virtual void _unbind(void* ptr, Handle* hndl) {
    // teehee
    update_hndl(hndl, nullptr);
    _bind(ptr, nullptr);
  }

  virtual bool _is_bound(void* ptr, Handle* hndl) {
    MetaHeader hdr = _read_header(ptr);
    // Check if region is active
    if (!hdr.is_active()) {
      // ERROR
    }
    return hdr.is_bound() && (hdr.hndl == hndl);;
  }

  virtual bool _has_handle(Handle* hndl) {
    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
      if(hdr_i->has_handle(hndl))
        return true;
    }
    return false;
  }

  virtual void* _allocate(size_t sz) {
    uint8_t* loc = nullptr;
    // If make this capacity then have possibility of filling up
    if (sz > size) {
      // ERROR
      return nullptr;
    }
    // if(sz > ( end() - (cursor + sizeof(MetaHeader)))){
    if (sz > available()) {
      // Allocate at beginning
      // Rebalance to make it less likely to overwrite a region
      // Overwriting allocated regions is a valid operation as long as the
      // overwritten regions are invalidated
      rebalance();
      // If still dont have space, error out
      if (sz > available()) {
        return nullptr;
      }
    }

    // First check to see if we have space in a previously allocated area
    // TODO: if this region is smaller split it and add another header to the table
    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
     if(!hdr_i->is_active() && hdr_i->get_len() >= sz){
       MetaHeader& hdr = *hdr_i;
       // Handle alignment
       void* aligned_loc = (void*) hdr._d;
       size_t space_change = hdr.get_len();
       aligned_loc = std::align(alignof(uint8_t*), sz, aligned_loc, space_change);
       hdr.set_active();
       //hdr.set_len(sz + hdr.get_len() - space_change);
       hdr.set_len(sz);
       hdr.set_hndl(nullptr);
       hdr.set_d((uint8_t*)aligned_loc);
       loc = (uint8_t*)aligned_loc;

       // Update capacity
       capacity -= hdr.get_len();

       return (void*)loc;
      
     }
    }
    
    
    // Otherwise allocate at the end
    MetaHeader hdr;
    // Handle alignment
    void* aligned_loc = (void*) cursor;
    size_t space_change = available();
    aligned_loc = std::align(alignof(uint8_t*), sz, aligned_loc, space_change);
    hdr.set_active();
    //hdr.set_len(sz + available() - space_change);
    hdr.set_len(sz);
    hdr.set_hndl(nullptr);
    hdr.set_d((uint8_t*)aligned_loc);
    _headers.push_back(hdr);
    loc = (uint8_t*)aligned_loc;
    cursor += hdr.get_len() + available() - space_change;

    // Update capacity
    capacity -= hdr.get_len();

    return (void*)loc;
  }

  virtual void _deallocate(void* ptr) {
    if (ptr) {
      MetaHeader& hdr = _read_header(ptr);
      hdr.set_inactive();
      if (hdr.is_bound()) {
        _unbind(ptr, hdr.hndl);
      }
      hdr.set_hndl(nullptr);  // cleanup
      capacity += hdr.get_len();
      // Do not update the size of the header
    }
  }

 public:
  localCircularArenaAllocator() : capacity(size) {
    memset(_buffer, 0, size);
    cursor = begin();
  }

  /** This implementation of rebalance shifts all allocated chunks to the end of
   * the buffer and inserts an inactive region at the start. note: cursor gets
   * moved to begin() note: unbound regions get wiped
   */
  //TODO Check to make sure updated locations are still aligned
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  //TODO ABOVE
  virtual bool rebalance() {
    // Clear all unbound entries
    for(auto hdr_i = _headers.rbegin(); hdr_i != _headers.rend(); hdr_i++){
      if(!hdr_i->is_bound()){
        hdr_i->set_inactive();  
      }
    }
    // Sort by activity (shifts unbound entries to the end)
    std::sort(_headers.begin(), _headers.end(), [](const MetaHeader& a, const MetaHeader& b) {
      return a.is_active() > b.is_active();
    });

    int pop_count = 0;
    for(auto hdr_i = _headers.rbegin(); hdr_i != _headers.rend(); hdr_i++){
      if(hdr_i->is_active()) {
        break;
      }
        capacity += hdr_i->get_len();
        pop_count++;
    }
    // Remove all unbound
    // Makes the allocator have a cold start
    for(int i = 0; i < pop_count; i++){
      _headers.pop_back();
    }

    
    // Headers now only has the bound regions
    // Sort by region
    std::sort(_headers.begin(), _headers.end(), [](const MetaHeader& a, const MetaHeader& b) {
      return a._d < b._d;
    });

    uint8_t tmp;
    cursor = begin();
    void* aligned_loc;
    size_t space_change;

    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
      aligned_loc = (void*) cursor;
      size_t space_change = available();
      aligned_loc = std::align(alignof(uint8_t*), hdr_i->get_len(), aligned_loc, space_change);

      // Shift the data
      for(int i = 0; i < hdr_i->get_len(); i++){
        tmp = hdr_i->_d[i];
        reinterpret_cast<uint8_t*>(aligned_loc)[i] = tmp;
      }

      // Update header
      //hdr_i->set_len(sz + available() - space_change);
      hdr_i->set_d((uint8_t*)aligned_loc);
      update_hndl(hdr_i->hndl, hdr_i->_d);
      cursor += hdr_i->get_len() + available() - space_change;
    }
    return true;
  }

  virtual size_t available() { return capacity; }

  virtual void clear() {
    // TODO deallocate and invalidate all references
    // reset to default state
    memset(_buffer, 0, size);
    cursor = begin();
    capacity = size;
  }

  // Check to see if pointer exists in memory space and is valid
  bool contains(void* p) const {
    if (!((p > _buffer) && (p < (_buffer + size)))) {
      return false;
    }
    for(auto hdr_i = _headers.begin(); hdr_i != _headers.end(); hdr_i++){
      if(hdr_i->_d == p)
        return hdr_i->is_used();
    }
    return false;
    //MetaHeader hdr = _read_header(p);
    //return hdr.is_used();
  }

 public:
  // Testing bits, attribute out later
  uint32_t internal_header_unit_size() const { return sizeof(MetaHeader); }
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
