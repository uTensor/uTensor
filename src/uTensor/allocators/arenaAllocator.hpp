#ifndef UTENSOR_ARENA_ALLOCATOR_HPP
#define UTENSOR_ARENA_ALLOCATOR_HPP
#include <cstdio>
#include <forward_list>
#include "memoryManagementInterface.hpp"
#include "tensor.hpp"

namespace uTensor {

#define BLOCK_INACTIVE (0 << 15)
#define BLOCK_ACTIVE (1 << 15)
#define BLOCK_ZERO_LENGTH 0

/**
 * Size allocated must be less than 2**15
 * TODO get around the BS alignment bits from the silly pointer variable causing
 * extra empty padding
 */
template <size_t size>
class localCircularArenaAllocator : public AllocatorInterface {
 private:
  class alignas(1) MetaHeader {
   public:
    uint16_t meta_data;
    Handle* hndl;

   public:
    MetaHeader()
        : meta_data(BLOCK_INACTIVE | BLOCK_ZERO_LENGTH), hndl(nullptr) {}
    MetaHeader(uint16_t sz) : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr) {}
    void set_active() { meta_data |= BLOCK_ACTIVE; }
    void set_inactive() { meta_data &= (BLOCK_INACTIVE | 0x7FFF); }
    void set_hndl(Handle* handle) { hndl = handle; }
    void set_len(uint16_t sz) {
      meta_data &= 0x8000;
      meta_data |= (0x7FFF & sz);
    }
    uint16_t get_len() const { return meta_data & 0x7FFF; }
    bool is_active() const { return (meta_data & 0x8000) == BLOCK_ACTIVE; }
    bool is_bound() const { return (hndl != nullptr); }
    bool has_handle(Handle* target) const {
      return is_active() && (hndl == target);
    }
    bool is_used() const { return is_active() && (get_len() > 0); }
  };

 private:
  uint16_t capacity;
  uint8_t _buffer[size];
  uint8_t* cursor;

  // Return the amount of free space at the tail
  uint16_t tail_capacity(){};

  MetaHeader _read_header(void* ptr) const {
    // First check if ptr in bounds
    if (ptr < _buffer || ptr > (_buffer + size)) {
      // ERROR
    }
    // Read the header
    uint8_t* p = reinterpret_cast<uint8_t*>(ptr);
    MetaHeader hdr;
    // TODO error check this
    memcpy(&hdr, p - sizeof(MetaHeader), sizeof(MetaHeader));
    return hdr;
  }
  void _write_header(const MetaHeader& hdr, void* ptr) {
    // First check if ptr in bounds
    if (ptr < _buffer || ptr > (_buffer + size)) {
      // ERROR
    }
    uint8_t* p = reinterpret_cast<uint8_t*>(ptr);
    memcpy(p - sizeof(MetaHeader), &hdr, sizeof(MetaHeader));
  }

  uint8_t* begin() { return _buffer + sizeof(MetaHeader); }
  uint8_t* end() { return _buffer + size; }
  void _clear_forward(size_t sz) {
    // clear necessary chunks until enough space for current request + appended
    // fragment header
    uint8_t* forward_cursor = cursor;
    while ((forward_cursor - cursor) < sz) {
      MetaHeader f_hdr = _read_header((void*)forward_cursor);
      deallocate((void*)forward_cursor);
      // Decide whether we need to insert a fragment header or not
      forward_cursor += f_hdr.get_len() + sizeof(MetaHeader);
      if ((forward_cursor - cursor) > (sz + sizeof(MetaHeader))) {
        f_hdr.set_inactive();
        f_hdr.set_hndl(nullptr);
        // set it to the free length
        f_hdr.set_len(forward_cursor - sizeof(MetaHeader) - cursor + sz);
        _write_header(f_hdr, (void*)(cursor + sz));
      }
    }
  }

 protected:
  virtual void _bind(void* ptr, Handle* hndl) {
    MetaHeader hdr = _read_header(ptr);
    // Check if region is active
    if (!hdr.is_active()) {
      // ERROR
    }
    hdr.set_hndl(hndl);
    _write_header(hdr, ptr);
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
    return hdr.is_bound();
  }

  virtual bool _has_handle(Handle* hndl) {
    uint8_t* m_cursor = begin();
    while (!(m_cursor > end())) {
      MetaHeader hdr = _read_header((void*)m_cursor);
      if (hdr.has_handle(hndl)) {
        return true;
      }
      if(hdr.get_len() == 0)
        break;
      m_cursor += hdr.get_len();
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
      // If still dont have space, start overwriting from the start
      if (sz > available()) {
        cursor = begin();
      }
    }
    MetaHeader hdr = _read_header((void*)cursor);
    if (hdr.is_active() || (hdr.get_len() > 0 && hdr.get_len() < sz)) {
      // clear necessary chunks until enough space for current request +
      // appended fragment header
      _clear_forward(sz);
    }
    hdr.set_active();
    hdr.set_len(sz);
    hdr.set_hndl(nullptr);
    _write_header(hdr, (void*)cursor);
    loc = cursor;
    // Cursor can corner case end up past the buffer region
    if ((cursor + hdr.get_len() + sizeof(MetaHeader)) > end()) {
      cursor = begin();
    } else {
      cursor += hdr.get_len() + sizeof(MetaHeader);
    }
    return (void*)loc;
  }
  virtual void _deallocate(void* ptr) {
    if (ptr) {
      MetaHeader hdr = _read_header(ptr);
      hdr.set_inactive();
      if (hdr.is_bound()) {
        _unbind(ptr, hdr.hndl);
      }
      hdr.set_hndl(nullptr);  // cleanup
      _write_header(hdr, ptr);
    }
  }

 public:
  localCircularArenaAllocator() : capacity(size) {
    printf("Sizeof Pointer = %d\n", sizeof(cursor));
    printf("Sizeof Handle = %d\n", sizeof(Handle));
    printf("MetaHeader Size = %d\n", sizeof(MetaHeader));
    memset(_buffer, 0, size);
    cursor = begin();
  }

  /** This implementation of rebalance shifts all allocated chunks to the end of
   * the buffer and inserts an inactive region at the start. note: cursor gets
   * moved to begin() note: unbound regions get wiped
   */
  virtual bool rebalance() {
    // TODO WARNING rebalancing Allocator
    // Shift each chunk towards the end of the buffer
    uint16_t empty_chunk_len;
    uint16_t allocated_amount;
    uint8_t* forward_cursor;
    uint8_t* fwrite_cursor;
    // First deallocate all unbound regions
    forward_cursor = begin();
    while (forward_cursor < end()) {
      MetaHeader hdr = _read_header((void*)forward_cursor);
      if (hdr.is_active() && !hdr.is_bound()) {
        deallocate((void*)forward_cursor);
      }
      forward_cursor += hdr.get_len() + sizeof(MetaHeader);
    }

    // Next shift all the bound regions to the start, forward scan
    // TODO do some sorting here to make smaller blocks at the start
    forward_cursor = begin();
    fwrite_cursor = begin();
    while (forward_cursor < end()) {
      MetaHeader hdr = _read_header((void*)forward_cursor);
      if (hdr.is_active() && hdr.is_bound() &&
          forward_cursor != fwrite_cursor) {
        memcpy(fwrite_cursor - sizeof(MetaHeader),
               forward_cursor - sizeof(MetaHeader),
               hdr.get_len() + sizeof(MetaHeader));
        fwrite_cursor += hdr.get_len() + sizeof(MetaHeader);
      }
      forward_cursor += hdr.get_len() + sizeof(MetaHeader);
    }
    cursor = fwrite_cursor;
    // Account for the extra empty meta data
    empty_chunk_len = (uint16_t)(end() - (cursor - sizeof(MetaHeader)));
    allocated_amount = (uint16_t)((cursor - sizeof(MetaHeader)) - _buffer);

    // From the end, move byte by byte until everything is shifted
    // TODO only move bound regions
    cursor = cursor - sizeof(MetaHeader) - 1;
    uint8_t* tail = &_buffer[size - 1];
    for (uint16_t i = 0; i < allocated_amount; i++) {
      *tail = *cursor;
      tail--;
      cursor--;
    }

    // Next scan forward from the shifted points and update any bound handles
    forward_cursor = _buffer + empty_chunk_len + sizeof(MetaHeader);
    while (forward_cursor < end()) {
      MetaHeader hdr = _read_header((void*)forward_cursor);
      if (hdr.is_bound()) {
        update_hndl(hdr.hndl, (void*)forward_cursor);
      }
      forward_cursor += hdr.get_len() + sizeof(MetaHeader);
    }

    // Write new header to start
    cursor = begin();
    MetaHeader hdr(empty_chunk_len);
    hdr.set_inactive();
    _write_header(hdr, (void*)cursor);
    return true;
  }

  virtual size_t available() { return end() - cursor; }

  virtual void clear() {
    // TODO deallocate and invalidate all references
    // reset to default state
    memset(_buffer, 0, size);
    cursor = begin();
  }

  // Check to see if pointer exists in memory space and is valid
  bool contains(void* p) const {
    if (!((p > _buffer) && (p < (_buffer + size)))) {
      return false;
    }
    MetaHeader hdr = _read_header(p);
    return hdr.is_used();
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
