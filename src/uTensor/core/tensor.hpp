#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"
#include "utensor_string.hpp"

namespace uTensor {
// Tensors also appear on the same heap as the Tensor metadata. This way we can move tensors around and delete them without affecting user code
//template <typename Allocator=utensor::DefaultTensorMetaDataAllocator>
class Tensor {
    private:
        utensor::TensorInterface* _ptr;
        // Cannot copy Tensors, must pass by reference
        Tensor(const Tensor& that);

    public:  
        utensor::TensorInterface* operator->(0); 
        Tensor(utensor::TensorInterface* ptr);
        // Add some bits to make the interface nicer to the user

        // Force everything to be on the utensor allocator
        void* operator new(size_t sz); 
        void operator delete(void* p); 

        // KEY BIT
        friend class utensor::AllocatorInterface;
};

// Same as Named Tensor but not registered in the context class 
struct SimpleNamedTensor {
    public:
    const uTensor::string& name; //Fixed
    Tensor& tensor;     //Modifiable
    
    SimpleNamedTensor(const uTensor::string& name, Tensor& tensor) : name(name), tensor(tensor);
};
}
#endif
