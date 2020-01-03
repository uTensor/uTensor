#include "RomTensor.hpp"
#include <cstdio>
namespace uTensor {
RomTensor::RomTensor(TensorShape _shape, ttype _type, const void* buffer) : 
    BufferTensor(_shape, _type, const_cast<void*>(buffer))
{ }

// TODO Need to fix the write/read selection functions
void* RomTensor::write(uint32_t linear_index) {
    //printf("[ERROR] Attempted write to ROM tensor\n");
    //return nullptr;
    return BufferTensor::write(linear_index);
}

RomTensor::~RomTensor() { }
void RomTensor::resize(TensorShape new_shape) {
    printf("[ERROR] Attempted resize of ROM tensor\n");
}
}
