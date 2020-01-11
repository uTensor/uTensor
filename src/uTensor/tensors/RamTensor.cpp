#include "RamTensor.hpp"
#include "context.hpp"

namespace uTensor {

size_t calc_required_space(const TensorShape& new_shape, uint8_t _type_size){
  return new_shape.get_linear_size() * _type_size;
}

RamTensor::RamTensor() {}
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
RamTensor::RamTensor(ttype _type)
    : TensorInterface(_type) {
}

RamTensor::RamTensor(TensorShape _shape, ttype _type)
    : TensorInterface(_shape, _type),
      _ram_region(Context::get_ram_data_allocator()->allocate(
          _shape.get_linear_size() * _type_size)) {
  AllocatorInterface* allocator = Context::get_ram_data_allocator();
  //bind(_ram_region, *allocator);
  allocator->bind(*_ram_region, &_ram_region);
}

RamTensor::~RamTensor() {
  Context::get_ram_data_allocator()->deallocate(*_ram_region);
}

void RamTensor::resize(TensorShape new_shape) {
  printf("Warning, RAM Tensor resize not implemented\n");
}

FutureMaxSizeRamTensor::FutureMaxSizeRamTensor(ttype _type) : RamTensor(_type), max_initial_size(0) {}
FutureMaxSizeRamTensor::FutureMaxSizeRamTensor(TensorShape _shape, ttype _type) :
                    RamTensor(_shape, _type) {
  max_initial_size = calc_required_space(_shape, _type_size);
}
FutureMaxSizeRamTensor::~FutureMaxSizeRamTensor() {}
void FutureMaxSizeRamTensor::resize(TensorShape new_shape) {
  if(max_initial_size == 0){
    _shape = new_shape;
    build();
    return;
  }
  if(calc_required_space(new_shape, _type_size) >= max_initial_size){
    printf("[ERROR] Resize of FutureTensor can only shrink\n");
    Context::get_default_context()->throwError(new InvalidResizeError());
    return;
  }
  _shape = new_shape;
}
void FutureMaxSizeRamTensor::build() { 
  _ram_region = Context::get_ram_data_allocator()->allocate( calc_required_space(_shape, _type_size));
  AllocatorInterface* allocator = Context::get_ram_data_allocator();
  //bind(_ram_region, *allocator);
  allocator->bind(*_ram_region, &_ram_region);
}

}  // namespace uTensor
