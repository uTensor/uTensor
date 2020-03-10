#include "tensorBase.hpp"

#include "context.hpp"
#include "memoryManagementInterface.hpp"
#include "uTensor_util.hpp"

namespace uTensor {
TensorBase::TensorBase() {
  Context::get_default_context()->register_tensor(this);
}
TensorBase::~TensorBase() {
  // Context::get_metadata_allocator()->deallocate(this);
}

// Allocate the tensor metadata on a different heap from the data scratch pads
void* TensorBase::operator new(size_t sz) {
  void* p =
      Context::get_default_context()->get_metadata_allocator()->allocate(sz);
  return p;
}

void TensorBase::operator delete(void* p) {
  Context::get_default_context()->get_metadata_allocator()->deallocate(p);
}

ttype TensorInterface::get_type() const { return _type; }
TensorShape& TensorInterface::get_shape() { return _shape; }
const TensorShape& TensorInterface::get_shape() const { return _shape; }
TensorInterface::TensorInterface()
    : TensorBase(), _shape(0), _type(undefined), _type_size(0) {}
TensorInterface::TensorInterface(ttype _type)
    : TensorBase(), _shape(0), _type(_type) {
  _type_size = type_size(_type);
}
TensorInterface::TensorInterface(TensorShape _shape, ttype _type)
    : TensorBase(), _shape(_shape), _type(_type) {
  _type_size = type_size(_type);
}
TensorInterface::~TensorInterface(){};

// Can access Tensors like
// mTensor(1) = 5, mTensor(2,2) = 5, etc.
const IntegralValue TensorInterface::operator()(uint16_t i, uint16_t j,
                                                uint16_t k, uint16_t l) const {
  // Add shape checks here
  return read(_shape.linear_index(i, j, k, l));
}
IntegralValue TensorInterface::operator()(uint16_t i, uint16_t j, uint16_t k,
                                          uint16_t l) {
  // Add shape checks here
  return write(_shape.linear_index(i, j, k, l));
}
const IntegralValue TensorInterface::operator()(uint32_t linear_index) const {
  // Add shape checks here
  return read(linear_index);
}
IntegralValue TensorInterface::operator()(uint32_t linear_index) {
  // Add shape checks here
  return write(linear_index);
}

TensorInterface& TensorInterface::set_quantization_params(
    const QuantizationParams& params) {
  _qnt_params = params;
  return *this;
}

const QuantizationParams& TensorInterface::get_quantization_params() const {
  return _qnt_params;
}

size_t TensorInterface::_get_readable_block(void* buffer,
                                            uint16_t req_read_size,
                                            uint32_t linear_index) const {
  uTensor_printf(
      "ERROR, Optimized op attempted to read access non-optimizable tensor\n");
  Context::get_default_context()->throwError(
      new InvalidOptimizableTensorError());
  return -1;
}
size_t TensorInterface::_get_writeable_block(void* buffer,
                                             uint16_t req_write_size,
                                             uint32_t linear_index) {
  uTensor_printf(
      "ERROR, Optimized op attempted to write access non-optimizable tensor\n");
  Context::get_default_context()->throwError(
      new InvalidOptimizableTensorError());
  return -1;
}
size_t TensorInterface::get_readable_block(void* buffer, uint16_t req_read_size,
                                           uint32_t linear_index) const {
  if (req_read_size > _type_size * _shape.get_linear_size()) {
    return -1;
  }
  return _get_readable_block(buffer, req_read_size, linear_index);
}
size_t TensorInterface::get_writeable_block(void* buffer,
                                            uint16_t req_write_size,
                                            uint32_t linear_index) {
  if (req_write_size > _type_size * _shape.get_linear_size()) {
    return -1;
  }
  return _get_writeable_block(buffer, req_write_size, linear_index);
}

}  // namespace uTensor
