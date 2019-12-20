#ifndef UTENSOR_ARENA_ALLOCATOR_HPP
#define UTENSOR_ARENA_ALLOCATOR_HPP
#include "memoryManagementInterface.hpp"
#include <forward_list>

namespace uTensor {

#define BLOCK_INACTIVE (0 << 15)
#define BLOCK_ACTIVE (1 << 15);
#define BLOCK_ZERO_LENGTH 0

/**
 * Size allocated must be less than 2**15
 */
template<size_t size>
class localCircularArenaAllocator : public AllocatorInterface {
  private:
    class MetaHeader {
        uint16_t meta_data;
        Handle* hndl;
      public:
        MetaHeader() : meta_data(BLOCK_INACTIVE | BLOCK_ZERO_LENGTH), hndl(nullptr) {}
        MetaHeader(uint16_t sz) : meta_data(BLOCK_ACTIVE | sz), hndl(nullptr) {}
        void set_active() { meta_data |= BLOCK_ACTIVE; }
        void set_inactive() { meta_data &= (BLOCK_ACTIVE | 0x7FFF) ; }
        void set_hndl(Handle* handle) { hndl = handle; }
        void set_len(uint16_t sz) { meta_data &= 0x8000; meta_data |= (0x7FFF & sz); }
        bool is_active() const { return (meta_data & 0x8000) == BLOCK_ACTIVE; }
        bool is_bound() const { return (hndl != nullptr); }
        bool has_handle(Handle* target) const { return hndl == target; }
    }
  private:
    uint16_t capacity;
    uint8_t _buffer[size];
    uint8_t* cursor;

    // Return the amount of free space at the tail
    uint16_t tail_capacity() { };

    MetaHeader _read_header(void* ptr) const {
      // First check if ptr in bounds
      if(ptr < _buffer || ptr > (_buffer + size)){
        //ERROR
      }
      // Read the header
      uint8_t* p = reinterpret_cast<uint8_t*>(ptr);
      MetaHeader hdr;
      // TODO error check this
      memcpy(&hdr, p - sizeof(MetaHeader), sizeof(MetaHeader));
      return hdr;
    }
    void _write_header(const MetaHeader& hdr, void* ptr){
      // First check if ptr in bounds
      if(ptr < _buffer || ptr > (_buffer + size)){
        //ERROR
      }
      uint8_t* p = reinterpret_cast<uint8_t*>(ptr);
      memcpy(p - sizeof(MetaHeader), &hdr, sizeof(MetaHeader));
    }
  protected:
    virtual void _bind(void* ptr, Handle* hndl){
      MetaHeader hdr = _read_header(ptr);
      //Check if region is active
      if(!hdr.is_active()){
        //ERROR
      }
      hdr.set_hndl(hndl);
      _write_header(hdr, ptr);
    }

    virtual void _unbind(void* ptr, Handle* hndl){
      //teehee
      _bind(ptr, nullptr);
    }

    virtual bool _is_bound(void* ptr, Handle* hndl){
      MetaHeader hdr = _read_header(ptr);
      //Check if region is active
      if(!hdr.is_active()){
        //ERROR
      }
      return hdr.is_bound();
    }

    virtual bool _has_handle(Handle* hndl) = 0;
    virtual void* _allocate(size_t sz) = 0;
    virtual void _deallocate(void* ptr) = 0;

  public:
    localCircularArenaAllocator() : capacity(size) {}

};

// Note not actually complete
template<size_t sz>
class ArenaCircularAllocator {
    private:
        static localCircularArenaAllocator<sz> _allocator;
    public:
        static void* allocate(size_t size) { 
            if (size > _allocator.available())
                return NULL;

            void* p = _allocator.allocate(size);
            if (p == NULL)
                _allocator.rebalance(); 
        }
        static void  deallocate(void* p) { ... }
        static void  bind(void* ptr, utensor::Tensor* hndl) {
            _allocator.bind(ptr, hndl);
        }

};

} //end namespace
#endif
