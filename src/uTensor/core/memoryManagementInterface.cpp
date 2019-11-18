#include "memoryManagementInterface.hpp"
#include "uTensor_util.hpp"

namespace uTensor {

void AllocatorInterface::update_hndl(Handle& h, void* new_ptr) {
    h._ptr = new_ptr;
}

void AllocatorInterface::bind(void* ptr, Handle* hndl) { 
    if (!_has_handle(hndl))
        DEBUG("Allocator does not contain reference to handle");

    if (is_bound(ptr, hndl)){
        ERR_EXIT("Cannot rebind Handles without unbinding");
    }
    _bind(ptr, hndl);
}
void AllocatorInterface::unbind(void* ptr, Handle* hndl) {
    if (!is_bound(ptr, hndl)){
        ERR_EXIT("Cannot unbind unbound Handles");
    }
    _unbind(ptr, hndl);            
}
bool AllocatorInterface::is_bound(void* ptr, Handle* hndl) {
    return _is_bound(ptr, hndl);
}

};
