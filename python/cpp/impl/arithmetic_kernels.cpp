#include "arithmetic_kernels.hpp"

#include "allocator.hpp"
#include "fast_copyop.hpp"
#include "uTensor/ops/Arithmetic_kernels.hpp"
#include "uTensor/core/types.hpp"

#include <array>

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
  // check whether dimensions are compatible to add
  if (info_a.ndim != info_b.ndim) {
    throw py::value_error("a and b should have the same number of dimensions");
  }
  for (int i = 0; i < info_a.ndim; i++) {
    if (info_a.shape[i] != info_b.shape[i]) {
      throw py::value_error("a and b should have the same shape");
    }
  }

  CopyOperator copy_op;
  // setup TensorShape, start with a 0-dim placeholder
  TensorShape shape(0);
  for (int i = 0; i < info_a.ndim; i++)
    shape[i] = info_a.shape[i];
  shape.update_dims();
  Tensor tensor_a = new RamTensor( shape, flt );
  Tensor tensor_b = new RamTensor( shape, flt );
  Tensor tensor_c = new RamTensor( shape, flt );
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
  // check whether dimensions are compatible to add
  if (info_a.ndim != info_b.ndim) {
    throw py::value_error("a and b should have the same number of dimensions");
  }
  for (int i = 0; i < info_a.ndim; i++) {
    if (info_a.shape[i] != info_b.shape[i]) {
      throw py::value_error("a and b should have the same shape");
    }
  }

  CopyOperator copy_op;
  // setup TensorShape, start with a 0-dim placeholder
  TensorShape shape(0);
  for (int i = 0; i < info_a.ndim; i++)
    shape[i] = info_a.shape[i];
  shape.update_dims();
  Tensor tensor_a = new RamTensor( shape, flt );
  Tensor tensor_b = new RamTensor( shape, flt );
  Tensor tensor_c = new RamTensor( shape, flt );
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
