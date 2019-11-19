#ifndef UTENSOR_ARENA_ALLOCATOR_HPP
#define UTENSOR_ARENA_ALLOCATOR_HPP
#include "memoryManagementInterface.hpp"
#include <forward_list>

namespace uTensor {
template<size_t size>
class localCircularArenaAllocator : public AllocatorInterface {
  private:
    uint16_t capacity;
    uint8_t _buffer[size];

    // Return the amount of free space at the tail
    uint16_t tail_capacity() { };
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
