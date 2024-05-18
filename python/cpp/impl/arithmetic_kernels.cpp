#include "arithmetic_kernels.hpp"

#include <array>

#include "allocator.hpp"
#include "fast_copyop.hpp"
#include "uTensor/core/types.hpp"
#include "uTensor/ops/Arithmetic_kernels.hpp"
#include "uTensor/ops/Broadcast.hpp"

namespace py = pybind11;
using uTensor::Context;
using uTensor::RamTensor;
using uTensor::Tensor;
using uTensor::python::get_meta_allocator;
using uTensor::python::get_ram_allocator;

py::array_t<float> add_kernel(const py::array_t<float> &a,
                              const py::array_t<float> &b) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_a = a.request();
  py::buffer_info info_b = b.request();

  CopyOperator copy_op;
  // setup TensorShape, start with a 0-dim placeholder
  TensorShape shape_a(0), shape_b(0), output_shape(0);
  for (int i = 0; i < info_a.ndim; i++) shape_a[i] = info_a.shape[i];
  for (int i = 0; i < info_b.ndim; i++) shape_b[i] = info_b.shape[i];
  shape_a.update_dims();
  shape_b.update_dims();
  if (!is_broadcastable(shape_a, shape_b, output_shape))
    throw py::value_error("a and b are not broadcastable");
  Tensor tensor_a = new RamTensor(shape_a, flt);
  Tensor tensor_b = new RamTensor(shape_b, flt);
  Tensor tensor_c = new RamTensor(output_shape, flt);
  copy_op.toTensor(info_a.ptr, tensor_a);
  copy_op.toTensor(info_b.ptr, tensor_b);
  uTensor::add_kernel<float>(tensor_c, tensor_a, tensor_b);
  py::buffer_info out_info = copy_op.getInfo(tensor_c);
  tensor_a.free();
  tensor_b.free();
  tensor_c.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);

  py::object base = py::cast(out_info.ptr);
  return py::array_t<float>(out_info, base);
}

py::array_t<float> mul_kernel(const py::array_t<float> &a,
                              const py::array_t<float> &b) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_a = a.request();
  py::buffer_info info_b = b.request();

  CopyOperator copy_op;
  // setup TensorShape, start with a 0-dim placeholder
  TensorShape shape_a(0), shape_b(0), output_shape(0);
  for (int i = 0; i < info_a.ndim; i++) shape_a[i] = info_a.shape[i];
  for (int i = 0; i < info_b.ndim; i++) shape_b[i] = info_b.shape[i];
  shape_a.update_dims();
  shape_b.update_dims();
  if (!is_broadcastable(shape_a, shape_b, output_shape))
    throw py::value_error("a and b are not broadcastable");
  Tensor tensor_a = new RamTensor(shape_a, flt);
  Tensor tensor_b = new RamTensor(shape_b, flt);
  Tensor tensor_c = new RamTensor(output_shape, flt);
  copy_op.toTensor(info_a.ptr, tensor_a);
  copy_op.toTensor(info_b.ptr, tensor_b);
  uTensor::mul_kernel<float>(tensor_c, tensor_a, tensor_b);
  py::buffer_info out_info = copy_op.getInfo(tensor_c);
  tensor_a.free();
  tensor_b.free();
  tensor_c.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);

  py::object base = py::cast(out_info.ptr);
  return py::array_t<float>(out_info, base);
}
