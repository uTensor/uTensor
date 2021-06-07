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
  const TensorShape& get_shape() const;

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
  SimpleNamedTensor();
  SimpleNamedTensor(const uTensor::string& name, Tensor& tensor);
  Tensor& tensor();

 public:
  const uTensor::string* name;  // Fixed
 private:
  Tensor* _tensor;  // Modifiable
};

class StridedIterator {
  /*
  https://gist.github.com/dboyliao/d4a72cbbd3f3e62517865519b4c1e9c6

  Given the slice specs, defined by the begin tensor, end tensor, strides tensor
  and its masks, StridedIterator will find all possible index tuples and compute
  the linear offset (i.e the index of flatten input)

  For example, if the given slice specs are 0:3 and 1:5:2, which give you the
  begin tensor [0, 1], end tensor [3, 5] and strides tensor [1, 2], and input
  tensor of shape (5, 10), all possible strided indices are:

  - (0, 1), with linear index 1
  - (0, 3), with linear index 3
  ....
  - (2, 9), with linear index 29
  */
 public:
  /*
    [NOTE] flat_*: ellipse, new axis and shrink masks are expanded
  */
  StridedIterator(const Tensor& input, const Tensor& flat_begin_tensor,
                  const Tensor& flat_end_tensor,
                  const Tensor& flat_strides_tensor, int32_t begin_mask = 0,
                  int32_t end_mask = 0);

  int32_t next();
  uint32_t num_elems() const;

 private:
  bool _hit_last;
  uint8_t _num_dims;
  uint32_t _num_elems;
  int32_t _idx_cnt[UTENSOR_MAX_NDIMS];
  int32_t _begin[UTENSOR_MAX_NDIMS];
  int32_t _end[UTENSOR_MAX_NDIMS];
  int32_t _strides[UTENSOR_MAX_NDIMS];
  uint32_t _in_strides[UTENSOR_MAX_NDIMS];
  int32_t _begin_mask;
  int32_t _end_mask;
  bool _is_done();
  void _reset();
};

}  // namespace uTensor
#endif
