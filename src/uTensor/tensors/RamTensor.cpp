#include "RamTensor.hpp"
#include "context.hpp"

namespace uTensor {

void* RamTensor::read(uint32_t linear_index) const {
  if (_ram_region) {
    uint8_t* d = reinterpret_cast<uint8_t*>(*_ram_region);
    return reinterpret_cast<void*>(d + linear_index * _type_size);
  }
  return nullptr;
}
void* RamTensor::write(uint32_t linear_index) {
  if (_ram_region) {
    uint8_t* d = reinterpret_cast<uint8_t*>(*_ram_region);
    return reinterpret_cast<void*>(d + linear_index * _type_size);
  }
  return nullptr;
}

RamTensor::RamTensor(TensorShape _shape, ttype _type)
    : TensorInterface(_shape, _type),
      _ram_region(Context::get_ram_data_allocator()->allocate(
          _shape.get_linear_size() * _type_size)) {
  AllocatorInterface* allocator = Context::get_ram_data_allocator();
  bind(_ram_region, *allocator);
}

RamTensor::~RamTensor() {
  Context::get_ram_data_allocator()->deallocate(*_ram_region);
}

void resize(TensorShape new_shape) {
  printf("Warning, RAM Tensor resize not implemented\n");
}

}  // namespace uTensor
