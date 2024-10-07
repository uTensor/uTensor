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
class Slice;
class alignas(alignof(uint8_t *)) Tensor : public Handle {
public:
  TensorInterface *operator->();
  const TensorInterface *operator->() const;
  // As long as operating on instantiations of this class and not pointers this
  // function will work
  TensorInterface *operator*();

  Tensor();
  Tensor(TensorInterface *ptr);
  Tensor &operator=(TensorInterface *ptr);
  Tensor(Tensor &&that);
  Tensor &operator=(Tensor &&that);
  ~Tensor();

  void free();
  Tensor view();
  Tensor view(const Slice &slice1);
  Tensor view(const Slice &slice1, const Slice &slice2);
  Tensor view(const Slice &slice1, const Slice &slice2, const Slice &slice3);
  Tensor view(const Slice &slice1, const Slice &slice2, const Slice &slice3,
              const Slice &slice4);

  // Add some bits to make the interface nicer to the user
  const IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                                 uint16_t l = 0) const;
  IntegralValue operator()(uint16_t i, uint16_t j, uint16_t k = 0,
                           uint16_t l = 0);
  const IntegralValue operator()(uint32_t linear_index) const;
  IntegralValue operator()(uint32_t linear_index);

  TensorShape &get_shape();
  const TensorShape &get_shape() const;

  // Force everything to be on the utensor allocator
  void *operator new(size_t sz);
  void operator delete(void *p);

  // KEY BIT
  friend class AllocatorInterface;

  // Add a couple of bits for GDB debugging since GDB doesnt support operator()
  IntegralValue gdb_read(uint16_t i);
  IntegralValue gdb_read(uint16_t i, uint16_t j);
  IntegralValue gdb_read(uint16_t i, uint16_t j, uint16_t k);
  IntegralValue gdb_read(uint16_t i, uint16_t j, uint16_t k, uint16_t l);
};

// Convenience
void print(const Tensor &t);

class TensorReference : public HandleReference {
public:
  TensorInterface *operator*();
};

// Same as Named Tensor but not registered in the context class
struct SimpleNamedTensor {
public:
  SimpleNamedTensor();
  SimpleNamedTensor(const uTensor::string &name, Tensor &tensor);
  Tensor &tensor();

public:
  const uTensor::string *name; // Fixed
private:
  Tensor *_tensor; // Modifiable
};

DECLARE_ERROR(ViewDoesNotSupportResizeError);

class Slice {
public:
  Slice(int32_t start, int32_t end, int32_t step = 1);

  inline int32_t start() const { return start_; }
  inline int32_t end() const { return end_; }
  inline int32_t step() const { return step_; }

private:
  int32_t start_, end_, step_;
};

class TensorView : public TensorInterface {
public:
  // NOTE: currently only support positive slicing
  // TODO: support negative indices and slicing
  // TODO: reshape
  TensorView(Tensor &other);
  TensorView(Tensor &other, const Slice &slice1);
  TensorView(Tensor &other, const Slice &slice1, const Slice &slice2);
  TensorView(Tensor &other, const Slice &slice1, const Slice &slice2,
             const Slice &slice3);
  TensorView(Tensor &other, const Slice &slice1, const Slice &slice2,
             const Slice &slice3, const Slice &slice4);

protected:
  virtual void *read(uint32_t linear_index) const override;
  virtual void *write(uint32_t linear_index) override;
  virtual void resize(const TensorShape &new_shape) override;
  virtual uint32_t compute_linear_index(uint16_t i, uint16_t j, uint16_t k = 0,
                                        uint16_t l = 0) const override;

private:
  Tensor &base_;
  int32_t _starts[4];
  int32_t _ends[4];
  int32_t _steps[4];
  void init_(int idx, const Slice &slice);
};
} // namespace uTensor
#endif
