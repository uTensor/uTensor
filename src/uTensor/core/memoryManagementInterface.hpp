#ifndef __UTENSOR_MEMORY_MANAGEMENT_IFC_H
#define __UTENSOR_MEMORY_MANAGEMENT_IFC_H
#include <cstring>
namespace uTensor {
class Tensor;

/**
 * Allocators are expected to maintain a mapping of Tensor handles to data regions. * This allows the allocator to move around the underlying data without breaking the user interface.
 */
class AllocatorInterface {

    // Allocators must implement these functions
    protected:
        virtual void _bind(void* ptr, Tensor* hndl) = 0;
        virtual void _unbind(void* ptr, Tensor* hndl) = 0;
        virtual bool _is_bound(void* ptr, Tensor* hndl) = 0;
        virtual bool _has_handle(Tensor* hndl) = 0;

    public:
        /*
         * Public interface for updating a Tensor Handle reference
         */
        void update_hndl(Tensor& h, Tensor* new_t_ptr);

        /**
         * Bind/Unbind data to Tensor Handle
         */
        void bind(void* ptr, Tensor* hndl);
        void unbind(void* ptr, Tensor* hndl); 
        /**
         * Check if a pointer is associated with a Tensor
         */
        bool is_bound(void* ptr, Tensor* hndl); 

        /**
         * Returns the amount of space available in the Memory Manager
         */
        virtual size_t available() = 0;

        /**
         * Update Tensor handles to point to new regions.
         * This is useful is the data moves around inside the memory manager,
         * For example if the data is compressed/decompressed dynamically
         */
        virtual bool rebalance() = 0; // KEY. This call updates all the Tensor data references
        
        /**
         * Allocate sz bytes in the memory manager
         */
        virtual void* allocate(size_t sz) = 0;

        /**
         * Deallocate all data associated with pointer
         */
        virtual void deallocate(void* ptr) = 0;

};


}
#endif
