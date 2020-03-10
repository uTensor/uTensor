#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"
#include "utensor_string.hpp"

namespace uTensor {
// Tensors also appear on the same heap as the Tensor metadata. This way we can
// move tensors around and delete them without affecting user code
// template <typename Allocator=utensor::DefaultTensorMetaDataAllocator>
//
class alignas(alignof(uint8_t*)) Tensor : public Handle {
 public:
  TensorInterface* operator->();
  const TensorInterface* operator->() const;
  // As long as operating on instantiations of this class and not pointers this
  // function will work
  TensorInterface* operator*();

  Tensor();
  Tensor(TensorInterface* ptr);
  Tensor& operator=(TensorInterface* ptr);
  Tensor(Tensor&& that);
  Tensor& operator=(Tensor&& that);
  ~Tensor();

  void free();

  // Add some bits to make the interface nicer to the user
  const IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                                 uint16_t l = 0) const;
  IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                           uint16_t l = 0);
  const IntegralValue operator()(uint32_t linear_index) const;
  IntegralValue operator()(uint32_t linear_index);

  TensorShape& get_shape();

  // Force everything to be on the utensor allocator
  void* operator new(size_t sz);
  void operator delete(void* p);

  // KEY BIT
  friend class AllocatorInterface;

  // Add a couple of bits for GDB debugging since GDB doesnt support operator()
  IntegralValue gdb_read(uint16_t i);
  IntegralValue gdb_read(uint16_t i, uint16_t j);
  IntegralValue gdb_read(uint16_t i, uint16_t j, uint16_t k);
  IntegralValue gdb_read(uint16_t i, uint16_t j, uint16_t k, uint16_t l);
};

// Convenience
void print(const Tensor& t);

class TensorReference : public HandleReference {
 public:
  TensorInterface* operator*();
};

// Same as Named Tensor but not registered in the context class
struct SimpleNamedTensor {
 public:
  const uTensor::string* name;  // Fixed
  Tensor* tensor;               // Modifiable

  SimpleNamedTensor();
  SimpleNamedTensor(const uTensor::string& name, Tensor& tensor);
};
}  // namespace uTensor
#endif
