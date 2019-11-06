#include "tensorBase.hpp"
namespace uTensor {
TensorBase::TensorBase(){
    utensor::Context::get_default_context().register(*this);
}

// Allocate the tensor metadata on a different heap from the data scratch pads
void* TensorBase::operator new(size_t sz) { 
    void* p = utensor::Context::DefaultTensorMetaDataAllocator::allocate(sz); 
    return p;
}

void TensorBase::operator delete(void* p) {
    utensor::Context::DefaultTensorMetaDataAllocator::deallocate(p);
}



ttype TensorInterface::get_type() const { return _type; }
TensorShape& TensorInterface::get_shape() { return _shape; }
TensorInterface::TensorInterface() : TensorBase(), _shape(0), _type(undefined) {}
TensorInterface::TensorInterface(TensorShape _shape, ttype _type) : TensorBase(), _shape(_shape), _type(_type) {}
virtual TensorInterface::~TensorInterface() {};

// Can access Tensors like
// mTensor(1) = 5, mTensor(2,2) = 5, etc.
const IntegralValue TensorInterface::operator()(uint16_t i, uint16_t j = 0, uint16_t k = 0, uint16_t l = 0){
    // Add shape checks here
    return read(_shape.linear_index(i, j, k, l));
}
IntegralValue& TensorInterface::operator()(uint16_t i, uint16_t j = 0, uint16_t k = 0, uint16_t l = 0){
    // Add shape checks here
    return write(_shape.linear_index(i,j,k,l));
}

}
