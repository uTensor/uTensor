#ifndef UTENSOR_RAM_TENSOR
#define UTENSOR_RAM_TENSOR

#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"

namespace uTensor {

size_t calc_required_space(const TensorShape& new_shape, uint8_t _type_size);

class RamTensor : public TensorInterface {
 protected:
  virtual void* read(uint32_t linear_index) const override;
  virtual void* write(uint32_t linear_index) override;
  RamTensor(); // May be useful in subclasses
  RamTensor(ttype _type);

 public:
  RamTensor(TensorShape _shape, ttype _type);
  virtual ~RamTensor();
  virtual void resize(TensorShape new_shape) override;

 protected:
  Handle _ram_region;
};

/**
 * Similar to RamTensor, but doesnt need to be allocated until its first resize
 */
class FutureMaxSizeRamTensor : public RamTensor {
  public:
    FutureMaxSizeRamTensor(ttype _type);
    FutureMaxSizeRamTensor(TensorShape _shape, ttype _type);
    virtual ~FutureMaxSizeRamTensor();
    // Resizing to something smaller only
    // Invalidates data
    virtual void resize(TensorShape new_shape) override;

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
