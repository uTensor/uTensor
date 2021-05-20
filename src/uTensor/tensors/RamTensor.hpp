#ifndef UTENSOR_RAM_TENSOR
#define UTENSOR_RAM_TENSOR

#include "uTensor/core/memoryManagementInterface.hpp"
#include "uTensor/core/tensorBase.hpp"

namespace uTensor {

size_t calc_required_space(const TensorShape& new_shape, uint8_t _type_size);

class RamTensor : public TensorInterface {
 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;
  RamTensor();  // May be useful in subclasses

 public:
  RamTensor(ttype _type);
  RamTensor(const TensorShape& _shape, ttype _type);
  virtual ~RamTensor();
  virtual void resize(const TensorShape& new_shape) override;
  // USE AT YOUR OWN RISK, this function is meant for testing purposes and should not be called elsewhere. It is likely to be removed or migrated to a test helper framework
  const void* get_address() { return *_ram_region; }

 protected:
  virtual size_t _get_readable_block(const void*& buffer, uint16_t req_read_size,
                                     uint32_t linear_index) const override;
  virtual size_t _get_writeable_block(void*& buffer, uint16_t req_write_size,
                                      uint32_t linear_index) override;

 protected:
  Handle _ram_region;
};

/**
 * Similar to RamTensor, but doesnt need to be allocated until its first resize
 */
class FutureMaxSizeRamTensor : public RamTensor {
 public:
  FutureMaxSizeRamTensor(ttype _type);
  FutureMaxSizeRamTensor(const TensorShape& _shape, ttype _type);
  virtual ~FutureMaxSizeRamTensor();
  // Resizing to something smaller only
  // Invalidates data
  virtual void resize(const TensorShape& new_shape) override;

 private:
  void build();

 private:
  size_t max_initial_size;
};

// class RawDataHandle : public Handle {
// public:
//  RawDataHandle(size_t req_ram_size) {
//    _ptr = Context::DefaultRamDataAllocator::allocate(req_ram_size);
//    Context::DefaultRamDataAllocator::bind(this, _ptr);
//  }
//  ~RawDataHandle() { Context::DefaultRamDataAllocator::deallocate(_ptr); }
//  // void* operator new(size_t sz) { // Have to delegate this size from
//  tensors
//  // somehow + sizeof(Tensor)
//  //  void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz);
//  //  return p;
//  //}
//  // void operator delete(void* p) {
//  //  Context::DefaultTensorMetaDataAllocator::deallocate(p);
//  //}
//  // Ignore this, use the constructor to specify the data space
//  //  void* operator new(size_t sz, size_t req_ram_size){
//  //    void* p = Context::DefaultTensorMetaDataAllocator::allocate(sz);
//  //    _ptr = Context::DefaultRamDataAllocator::allocator(req_ram_size);
//  //    return p;
//  //  }
//};

}  // namespace uTensor
#endif
