#include "uTensor/core/tensor.hpp"

#include "uTensor/core/context.hpp"
#include "uTensor/core/uTensor_util.hpp"

namespace uTensor {
// Tensor::Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by
// reference

TensorInterface* Tensor::operator->() {
  return reinterpret_cast<TensorInterface*>(_ptr);
}
const TensorInterface* Tensor::operator->() const {
  return reinterpret_cast<const TensorInterface*>(_ptr);
}
TensorInterface* Tensor::operator*() {
  return reinterpret_cast<TensorInterface*>(_ptr);
}
Tensor::~Tensor() { free(); }
void Tensor::free() {
  void* ptr_t = _ptr;  // unbind invalidates this handle so store a copy
  if (_ptr) {
    AllocatorInterface* alloc =
        Context::get_default_context()->get_metadata_allocator();
    if (alloc->is_bound(_ptr, this)) {
      alloc->unbind(_ptr, this);
    }
    delete reinterpret_cast<TensorInterface*>(ptr_t);
    alloc->deallocate(ptr_t);
  }
  _ptr = nullptr;
}
Tensor::Tensor() : Handle() {}
Tensor::Tensor(TensorInterface* ptr) : Handle((void*)ptr) {
  // Context::get_default_context()->get_metadata_allocator()->bind(_ptr, this);
  if (ptr != nullptr) {
    bind(*this, *Context::get_default_context()->get_metadata_allocator());
  }
}
Tensor& Tensor::operator=(TensorInterface* ptr) {
  _ptr = (void*)ptr;
  bind(*this, *Context::get_default_context()->get_metadata_allocator());
  // Context::get_metadata_allocator()->bind(_ptr, this);
  return *this;
}

Tensor::Tensor(Tensor&& that) {
  _ptr = that._ptr;
  AllocatorInterface* alloc =
      Context::get_default_context()->get_metadata_allocator();
  if (alloc->is_bound(_ptr, &that)) {
    alloc->unbind(_ptr, &that);
    alloc->bind(_ptr, this);
  }
  that._ptr = nullptr;
}
Tensor& Tensor::operator=(Tensor&& that) {
  if (this != &that) {
    _ptr = that._ptr;
    AllocatorInterface* alloc =
        Context::get_default_context()->get_metadata_allocator();
    if (alloc->is_bound(_ptr, &that)) {
      alloc->unbind(_ptr, &that);
      alloc->bind(_ptr, this);
    }
    that._ptr = nullptr;
  }
  return *this;
}
// Add some bits to make the interface nicer to the user

// Force everything to be on the utensor allocator
void* Tensor::operator new(size_t sz) {  // Have to delegate this size from
                                         // tensors somehow + sizeof(Tensor)
  void* p =
      Context::get_default_context()->get_metadata_allocator()->allocate(sz);
  return p;
}
void Tensor::operator delete(void* p) {
  Context::get_default_context()->get_metadata_allocator()->deallocate(p);
}

// Interface
const IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                                       uint16_t l) const {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k, l);
}
IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                                 uint16_t l) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k, l);
}
const IntegralValue Tensor::operator()(uint32_t linear_index) const {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(linear_index);
}
IntegralValue Tensor::operator()(uint32_t linear_index) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(linear_index);
}

TensorShape& Tensor::get_shape() {
  return reinterpret_cast<TensorInterface*>(_ptr)->get_shape();
}

const TensorShape& Tensor::get_shape() const {
  return reinterpret_cast<TensorInterface*>(_ptr)->get_shape();
}

TensorInterface* TensorReference::operator*() {
  return reinterpret_cast<TensorInterface*>(_ref->operator*());
}

// Add a couple of bits for GDB debugging since GDB doesnt support operator()
IntegralValue Tensor::gdb_read(uint16_t i) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j, uint16_t k) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
  return reinterpret_cast<TensorInterface*>(_ptr)->operator()(i, j, k, l);
}

void print(const Tensor& t) {
  const TensorShape& t_shape = t->get_shape();
  if (t_shape.num_dims() > 2) {
    uTensor_printf("printing > 2D tensors not supported\n");
    return;
  }
  uTensor_printf("[\n");
  for (int j = 0; j < t_shape[1]; j++) {
    uTensor_printf("[ ");
    for (int i = 0; i < t_shape[0]; i++) {
      switch (t->get_type()) {
        case u8:
          uTensor_printf("%hhu", static_cast<uint8_t>(t(j, i)));
          break;
        case i8:
          uTensor_printf("%hhd", static_cast<int8_t>(t(j, i)));
          break;
        case u16:
          uTensor_printf("%hu", static_cast<uint16_t>(t(j, i)));
          break;
        case i16:
          uTensor_printf("%hd", static_cast<int16_t>(t(j, i)));
          break;
        case u32:
          uTensor_printf("%u", static_cast<uint32_t>(t(j, i)));
          break;
        case i32:
          uTensor_printf("%d", static_cast<int32_t>(t(j, i)));
          break;
        case flt:
          uTensor_printf("%f", static_cast<float>(t(j, i)));
          break;
        default:
          uTensor_printf("Unknown data type");
          return;
      }
      if (i != (t_shape[0] - 1)) {
        uTensor_printf(", ");
      } else {
        uTensor_printf(" ");
      }
    }
    uTensor_printf("]\n");
  }
  uTensor_printf("]\n");
}

SimpleNamedTensor::SimpleNamedTensor(const uTensor::string& name,
                                     Tensor& tensor)
    : name(&name), _tensor(&tensor) {}
SimpleNamedTensor::SimpleNamedTensor() : name(nullptr), _tensor(nullptr) {}
Tensor& SimpleNamedTensor::tensor() { return *_tensor; }

StridedIterator::StridedIterator(const Tensor& input,
                                 const Tensor& flat_begin_tensor,
                                 const Tensor& flat_end_tensor,
                                 const Tensor& flat_strides_tensor,
                                 int32_t begin_mask, int32_t end_mask)
    : _hit_last(false),
      _num_elems(1),
      _begin_mask(begin_mask),
      _end_mask(end_mask) {
  /*
    IMPORTANT: StridedIterator handels only flat inputs, i.e, no new axis, no
    ellipse and no shrink
   */
  // TODO: add runtime checks on flat inputs
  _num_dims = input.get_shape().num_dims();
  uint32_t idx_num = flat_begin_tensor.get_shape().num_elems();
  TensorStrides in_strides(input.get_shape());
  const TensorShape& in_shape = input.get_shape();
  for (uint32_t i = 0; i < idx_num; ++i) {
    uint16_t dim_size = in_shape[i];
    int32_t begin_idx = flat_begin_tensor(i);
    if (begin_idx < 0) begin_idx += dim_size;
    if ((1 << i) & _begin_mask) begin_idx = 0;
    int32_t end_idx = flat_end_tensor(i);
    if (end_idx < 0) end_idx += dim_size;
    if ((1 << i) & _end_mask) end_idx = dim_size;
    int32_t stride = flat_strides_tensor(i);
    _idx_cnt[i] = begin_idx;
    _begin[i] = begin_idx;
    _end[i] = end_idx;
    _strides[i] = stride;
    _in_strides[i] = in_strides[i];
  }
  for (uint32_t i = idx_num; i < _num_dims; ++i) {
    _idx_cnt[i] = 0;
    _begin[i] = 0;
    _end[i] = in_shape[i];
    _strides[i] = 1;
  }

  for (size_t i = 0; i < _num_dims; ++i) {
    int32_t e = _end[i];
    int32_t s = _strides[i];
    int32_t cnt = 1;
    int32_t acc = _begin[i] + s;
    while (acc < e) {
      cnt += 1;
      acc += s;
    }
    _num_elems *= cnt;
  }
}

bool StridedIterator::_is_done() {
  bool done = true;
  size_t i = 0;
  for (size_t i = 0; i < _num_dims; ++i) {
    if (_idx_cnt[i] < _end[i]) {
      done = false;
      break;
    }
  }
  return done;
}

void StridedIterator::_reset() {
  _hit_last = false;
  for (size_t i = 0; i < _num_dims; ++i) {
    _idx_cnt[i] = _begin[i];
  }
}

int32_t StridedIterator::next() {
  // compute linear offset at the moment
  int32_t linear_offset = 0;
  for (size_t i = 0; i < _num_dims; ++i) {
    linear_offset += _idx_cnt[i] * _in_strides[i];
  }
  // update indices
  _idx_cnt[_num_dims - 1] += _strides[_num_dims - 1];
  for (int i = _num_dims - 2; i >= 0; --i) {
    if (_idx_cnt[i + 1] >= _end[i + 1]) {
      _idx_cnt[i] += _strides[i];
    }
  }
  // check if it's done
  bool done = _is_done();
  if (_hit_last || done) {
    if (_hit_last) {
      _reset();
      return -1;
    } else {
      _hit_last = true;
    }
  };
  // check if the idx exceed end
  for (size_t i = 0; i < _num_dims; ++i) {
    if (_idx_cnt[i] >= _end[i]) {
      _idx_cnt[i] = _begin[i];
    }
  }
  return linear_offset;
}

uint32_t StridedIterator::num_elems() const { return _num_elems; }
}  // namespace uTensor
