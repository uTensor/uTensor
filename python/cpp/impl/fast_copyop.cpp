#include "fast_copyop.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

using uTensor::Tensor;

void CopyOperator::toTensor(const void *src, Tensor &dest) {
  void *dest_buffer;
  auto num_elems = dest->num_elems();
  size_t count = static_cast<size_t>(type_size(dest->get_type())) *
                 static_cast<size_t>(num_elems);
  get_writeable_block(dest, dest_buffer, num_elems, 0);
  std::memcpy(dest_buffer, src, count);
}

void CopyOperator::fromTensor(void *dest, const Tensor &src) {
  const void *src_buffer;
  auto num_elems = src->num_elems();
  size_t count = static_cast<size_t>(type_size(src->get_type())) *
                 static_cast<size_t>(num_elems);
  get_readable_block(src, src_buffer, num_elems, 0);
  std::memcpy(dest, src_buffer, count);
}

py::buffer_info CopyOperator::getInfo(const Tensor &tensor) {
  py::buffer_info info;
  uint32_t num_elems = tensor->num_elems();
  py::ssize_t item_size =
      static_cast<py::ssize_t>(type_size(tensor->get_type()));
  void *src_buffer = std::malloc(static_cast<size_t>(item_size) *
                                 static_cast<size_t>(num_elems));
  fromTensor(src_buffer, tensor);

  std::string fmt;
  switch (tensor->get_type()) {
    case flt:
      fmt = py::format_descriptor<float>::format();
      break;
    case i8:
      fmt = py::format_descriptor<int8_t>::format();
      break;
    case u8:
      fmt = py::format_descriptor<uint8_t>::format();
      break;
    case i16:
      fmt = py::format_descriptor<int16_t>::format();
      break;
    case u16:
      fmt = py::format_descriptor<u_int16_t>::format();
      break;
    case i32:
      fmt = py::format_descriptor<int32_t>::format();
      break;
    case u32:
      fmt = py::format_descriptor<u_int32_t>::format();
      break;
    default:
      throw py::type_error("unknown tensor data type");
      break;
  }
  TensorShape shape = tensor->get_shape();
  py::ssize_t ndims = static_cast<py::ssize_t>(shape.num_dims());

  std::vector<py::ssize_t> v_shape(ndims);
  std::vector<py::ssize_t> v_strides(ndims);
  py::ssize_t s = 1;
  for (size_t idx = 0; idx < ndims; ++idx) {
    v_shape.at(idx) = shape[idx];
    v_strides.at(ndims - idx - 1) = s * item_size;
    s *= shape[ndims - idx - 1];
  }
  return py::buffer_info(
      src_buffer, item_size, fmt, ndims,
      py::detail::any_container<py::ssize_t>(v_shape.begin(), v_shape.end()),
      py::detail::any_container<py::ssize_t>(v_strides.begin(),
                                             v_strides.end()));
}