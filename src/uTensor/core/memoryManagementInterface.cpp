#include "memoryManagementInterface.hpp"
#include "uTensor_util.hpp"

namespace uTensor {

void AllocatorInterface::update_hndl(Tensor& h, Tensor* new_t_ptr) {
    h._ptr = new_t_ptr;
}

void AllocatorInterface::bind(void* ptr, Tensor* hndl) { 
    if (!has_hndl(hndl))
        DEBUG("Allocator does not contain reference to handle");

    if (is_bound(ptr, hndl)){
        ERR_EXIT("Cannot rebind Handles without unbinding");
    }
    _bind(ptr, hndl);
}
void AllocatorInterface::unbind(void* ptr, Tensor* hdnl) {
    if (!is_bound(ptr, hndl)){
        ERR_EXIT("Cannot unbind unbound Handles");
    }
    _unbind(ptr, hndl);            
}
bool AllocatorInterface::is_bound(void* ptr, Tensor* hndl) {
    return _is_bound(ptr, hndl);
}

};
