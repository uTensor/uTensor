#include "uTensor/core/tensor.hpp"

#include "uTensor/core/context.hpp"
#include "uTensor/core/uTensor_util.hpp"

namespace uTensor {
// Tensor::Tensor(const Tensor& that) {} // Cannot copy Tensors, must pass by
// reference

TensorInterface *Tensor::operator->() {
  return reinterpret_cast<TensorInterface *>(_ptr);
}
const TensorInterface *Tensor::operator->() const {
  return reinterpret_cast<const TensorInterface *>(_ptr);
}
TensorInterface *Tensor::operator*() {
  return reinterpret_cast<TensorInterface *>(_ptr);
}
Tensor::~Tensor() { free(); }
void Tensor::free() {
  void *ptr_t = _ptr; // unbind invalidates this handle so store a copy
  if (_ptr) {
    AllocatorInterface *alloc =
        Context::get_default_context()->get_metadata_allocator();
    if (alloc->is_bound(_ptr, this)) {
      alloc->unbind(_ptr, this);
    }
    delete reinterpret_cast<TensorInterface *>(ptr_t);
    alloc->deallocate(ptr_t);
  }
  _ptr = nullptr;
}
Tensor::Tensor() : Handle() {}
Tensor::Tensor(TensorInterface *ptr) : Handle((void *)ptr) {
  // Context::get_default_context()->get_metadata_allocator()->bind(_ptr, this);
  if (ptr != nullptr) {
    bind(*this, *Context::get_default_context()->get_metadata_allocator());
  }
}
Tensor &Tensor::operator=(TensorInterface *ptr) {
  _ptr = (void *)ptr;
  bind(*this, *Context::get_default_context()->get_metadata_allocator());
  // Context::get_metadata_allocator()->bind(_ptr, this);
  return *this;
}

Tensor::Tensor(Tensor &&that) {
  _ptr = that._ptr;
  AllocatorInterface *alloc =
      Context::get_default_context()->get_metadata_allocator();
  if (alloc->is_bound(_ptr, &that)) {
    alloc->unbind(_ptr, &that);
    alloc->bind(_ptr, this);
  }
  that._ptr = nullptr;
}
Tensor &Tensor::operator=(Tensor &&that) {
  if (this != &that) {
    _ptr = that._ptr;
    AllocatorInterface *alloc =
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
void *Tensor::operator new(size_t sz) { // Have to delegate this size from
                                        // tensors somehow + sizeof(Tensor)
  void *p =
      Context::get_default_context()->get_metadata_allocator()->allocate(sz);
  return p;
}
void Tensor::operator delete(void *p) {
  Context::get_default_context()->get_metadata_allocator()->deallocate(p);
}

// Interface
const IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                                       uint16_t l) const {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i, j, k, l);
}
IntegralValue Tensor::operator()(uint16_t i, uint16_t j, uint16_t k,
                                 uint16_t l) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i, j, k, l);
}
const IntegralValue Tensor::operator()(uint32_t linear_index) const {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(linear_index);
}
IntegralValue Tensor::operator()(uint32_t linear_index) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(linear_index);
}

TensorShape &Tensor::get_shape() {
  return reinterpret_cast<TensorInterface *>(_ptr)->get_shape();
}

const TensorShape &Tensor::get_shape() const {
  return reinterpret_cast<TensorInterface *>(_ptr)->get_shape();
}

TensorInterface *TensorReference::operator*() {
  return reinterpret_cast<TensorInterface *>(_ref->operator*());
}

// Add a couple of bits for GDB debugging since GDB doesnt support operator()
IntegralValue Tensor::gdb_read(uint16_t i) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i, j);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j, uint16_t k) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i, j, k);
}
IntegralValue Tensor::gdb_read(uint16_t i, uint16_t j, uint16_t k, uint16_t l) {
  return reinterpret_cast<TensorInterface *>(_ptr)->operator()(i, j, k, l);
}

void print(const Tensor &t) {
  const TensorShape &t_shape = t->get_shape();
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

SimpleNamedTensor::SimpleNamedTensor(const uTensor::string &name,
                                     Tensor &tensor)
    : name(&name), _tensor(&tensor) {}
SimpleNamedTensor::SimpleNamedTensor() : name(nullptr), _tensor(nullptr) {}
Tensor &SimpleNamedTensor::tensor() { return *_tensor; }

DEFINE_ERROR(ViewDoesNotSupportResizeError);

Slice::Slice(int32_t start, int32_t end, int32_t step)
    : start_(start), end_(end), step_(step) {}

TensorView::TensorView(Tensor &other)
    : base_(other), TensorInterface(other->get_shape(), other->get_type()) {}

void TensorView::init_(int idx, const Slice &slice) {
  _starts[idx] = slice.start();
  _ends[idx] = slice.end();
  _steps[idx] = slice.step();
  _shape[idx] = (slice.end() - slice.start()) / slice.step() + 1;
}

TensorView::TensorView(Tensor &other, const Slice &slice1) : TensorView(other) {
  init_(0, slice1);
}

TensorView::TensorView(Tensor &other, const Slice &slice1, const Slice &slice2)
    : TensorView(other, slice1) {
  init_(1, slice2);
}

TensorView::TensorView(Tensor &other, const Slice &slice1, const Slice &slice2,
                       const Slice &slice3)
    : TensorView(other, slice1, slice2) {
  init_(2, slice3);
}
TensorView::TensorView(Tensor &other, const Slice &slice1, const Slice &slice2,
                       const Slice &slice3, const Slice &slice4)
    : TensorView(other, slice1, slice2, slice3) {
  init_(3, slice4);
}

uint32_t TensorView::compute_linear_index(uint16_t i, uint16_t j, uint16_t k,
                                          uint16_t l) const {
  return base_->get_shape().linear_index(
      i * _steps[0] + _starts[0], j * _steps[1] + _starts[1],
      k * _steps[2] + _starts[2], l * _steps[3] + _starts[3]);
}

void *TensorView::read(uint32_t linear_index) const {
  return base_->read(linear_index);
}

void *TensorView::write(uint32_t linear_index) {
  return base_->write(linear_index);
}

void TensorView::resize(const TensorShape &new_shape) {
  Context::get_default_context()->throwError(new ViewDoesNotSupportResizeError);
}

Tensor Tensor::view() { return Tensor(new TensorView(*this)); }
Tensor Tensor::view(const Slice &slice1) {
  return Tensor(new TensorView(*this, slice1));
}
Tensor Tensor::view(const Slice &slice1, const Slice &slice2) {
  return Tensor(new TensorView(*this, slice1, slice2));
}
Tensor Tensor::view(const Slice &slice1, const Slice &slice2,
                    const Slice &slice3) {
  return Tensor(new TensorView(*this, slice1, slice2, slice3));
}
Tensor Tensor::view(const Slice &slice1, const Slice &slice2,
                    const Slice &slice3, const Slice &slice4) {
  return Tensor(new TensorView(*this, slice1, slice2, slice3, slice4));
}

} // namespace uTensor
