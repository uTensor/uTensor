#include "BufferTensor.hpp"
#include "context.hpp"
#include <cstdio>

namespace uTensor {

void BufferTensor::resize(TensorShape new_shape) {
    printf("[ERROR] Cannot resize a BufferTensor\n");
    Context::get_default_context()->throwError(new InvalidResizeError());
}

BufferTensor::BufferTensor(TensorShape _shape, ttype _type) : TensorInterface(_shape, _type), _buffer(nullptr) {}
BufferTensor::BufferTensor(TensorShape _shape, ttype _type, void* buffer) : TensorInterface(_shape, _type), _buffer(reinterpret_cast<uint8_t*>(buffer)) {}
BufferTensor::~BufferTensor(){ _buffer = nullptr; }

bool BufferTensor::is_bound_to_buffer() const { return _buffer != nullptr; }
bool BufferTensor::is_bound_to_buffer(void* b) const { return (_buffer != nullptr) && (_buffer == b); }
bool BufferTensor::bind(void* b) {
    if(!is_bound_to_buffer()) {
        _buffer = reinterpret_cast<uint8_t*>(b);
        return true;
    }
    return false;
}
bool BufferTensor::unbind() {
    _buffer = nullptr;
    return true;
}

void* BufferTensor::read(uint32_t linear_index) const {
  if (_buffer) {
    //uint8_t* d = reinterpret_cast<uint8_t*>(_buffer);
    return reinterpret_cast<void*>(_buffer + linear_index * _type_size);
  }
  return nullptr;
}
void* BufferTensor::write(uint32_t linear_index) {
  if (_buffer) {
    //uint8_t* d = reinterpret_cast<uint8_t*>(*_buffer);
    return reinterpret_cast<void*>(_buffer + linear_index * _type_size);
  }
  return nullptr;
}
}
