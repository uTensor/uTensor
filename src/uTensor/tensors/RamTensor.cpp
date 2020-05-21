#include "RamTensor.hpp"

#include "context.hpp"
#include "uTensor_util.hpp"

namespace uTensor {

size_t calc_required_space(const TensorShape& new_shape, uint8_t _type_size) {
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

size_t RamTensor::_get_readable_block(void* buffer, uint16_t req_read_size,
                                      uint32_t linear_index) const {
  if (req_read_size + linear_index > _type_size * _shape.get_linear_size()) {
    Context::get_default_context()->throwError(new InvalidMemAccessError());
    return -1;
  }
  uint8_t* d = reinterpret_cast<uint8_t*>(*_ram_region);
  buffer = reinterpret_cast<void*>(d + linear_index * _type_size);
  return req_read_size;
}
size_t RamTensor::_get_writeable_block(void* buffer, uint16_t req_write_size,
                                       uint32_t linear_index) {
  if (req_write_size + linear_index > _type_size * _shape.get_linear_size()) {
    Context::get_default_context()->throwError(new InvalidMemAccessError());
    return -1;
  }
  uint8_t* d = reinterpret_cast<uint8_t*>(*_ram_region);
  buffer = reinterpret_cast<void*>(d + linear_index * _type_size);
  return req_write_size;
}

RamTensor::RamTensor(ttype _type) : TensorInterface(_type) {}

RamTensor::RamTensor(TensorShape _shape, ttype _type)
    : TensorInterface(_shape, _type),
      _ram_region(
          Context::get_default_context()->get_ram_data_allocator()->allocate(
              _shape.get_linear_size() * _type_size)) {
  AllocatorInterface* allocator =
      Context::get_default_context()->get_ram_data_allocator();
  // bind(_ram_region, *allocator);
  allocator->bind(*_ram_region, &_ram_region);
}

RamTensor::~RamTensor() {
  AllocatorInterface* alloc = Context::get_default_context()->get_ram_data_allocator();
  //alloc->unbind_and_deallocate(&_ram_region);
  void* ptr_t = *_ram_region;
    if (alloc->is_bound(*_ram_region, &_ram_region)) {
      alloc->unbind(*_ram_region, &_ram_region);
    }
    alloc->deallocate(ptr_t);
}

void RamTensor::resize(TensorShape new_shape) {
  AllocatorInterface* allocator = Context::get_default_context()->get_ram_data_allocator();
  // unbind handle before reallocate memory
  void* old_ptr = *_ram_region;
  if (is_bound(_ram_region, *allocator)) {
    allocator->unbind(*_ram_region, &_ram_region);
  }
  if(old_ptr)
    allocator->deallocate(old_ptr);
  void* ptr = allocator->allocate(new_shape.get_linear_size()*_type_size);
  if (!ptr) {
    uTensor_printf("OOM when resizing\n");
    Context::get_default_context()->throwError(new OutOfMemError);
    return;
  }
  _shape = new_shape;
  allocator->bind(ptr, &_ram_region);
}

FutureMaxSizeRamTensor::FutureMaxSizeRamTensor(ttype _type)
    : RamTensor(_type), max_initial_size(0) {}
FutureMaxSizeRamTensor::FutureMaxSizeRamTensor(TensorShape _shape, ttype _type)
    : RamTensor(_shape, _type) {
  max_initial_size = calc_required_space(_shape, _type_size);
}
FutureMaxSizeRamTensor::~FutureMaxSizeRamTensor() {}
void FutureMaxSizeRamTensor::resize(TensorShape new_shape) {
  if (max_initial_size == 0) {
    _shape = new_shape;
    build();
    return;
  }
  if (calc_required_space(new_shape, _type_size) >= max_initial_size) {
    uTensor_printf("[ERROR] Resize of FutureTensor can only shrink\n");
    Context::get_default_context()->get_default_context()->throwError(
        new InvalidResizeError());
    return;
  }
  _shape = new_shape;
}
void FutureMaxSizeRamTensor::build() {
  _ram_region =
      Context::get_default_context()->get_ram_data_allocator()->allocate(
          calc_required_space(_shape, _type_size));
  AllocatorInterface* allocator =
      Context::get_default_context()->get_ram_data_allocator();
  // bind(_ram_region, *allocator);
  allocator->bind(*_ram_region, &_ram_region);
}

}  // namespace uTensor
