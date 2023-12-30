#include <cstddef>

#include "arenaAllocator.hpp"
#include "play_fastop.hpp"
#include "pybind11/pybind11.h"

namespace py = pybind11;
using uTensor::Context;
using uTensor::RamTensor;
using uTensor::Tensor;
using uTensor::ReferenceOperators::MatrixMultOperatorV2;

static uTensor::localCircularArenaAllocator<1024> meta_allocator;
static uTensor::localCircularArenaAllocator<1024> ram_allocator;

py::array_t<float> matmul(const py::array_t<float> &a,
                          const py::array_t<float> &b) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  py::buffer_info info_a = a.request();
  py::buffer_info info_b = b.request();
  if (info_a.ndim != 2 || info_b.ndim != 2) {
    throw py::value_error("a and b should be both 2-dims array");
  }
  CopyOperator copy_op;
  MatrixMultOperatorV2<float> matmul_op;
  Tensor tensor_a = new RamTensor({static_cast<uint16_t>(info_a.shape[0]),
                                   static_cast<uint16_t>(info_a.shape[1])},
                                  flt);
  Tensor tensor_b = new RamTensor({static_cast<uint16_t>(info_b.shape[0]),
                                   static_cast<uint16_t>(info_b.shape[1])},
                                  flt);
  Tensor tensor_c = new RamTensor({static_cast<uint16_t>(info_a.shape[0]),
                                   static_cast<uint16_t>(info_b.shape[1])},
                                  flt);
  copy_op.toTensor(info_a.ptr, tensor_a);
  copy_op.toTensor(info_b.ptr, tensor_b);
  matmul_op
      .set_inputs({
          {MatrixMultOperatorV2<float>::input, tensor_a},
          {MatrixMultOperatorV2<float>::filter, tensor_b},
      })
      .set_outputs({
          {MatrixMultOperatorV2<float>::output, tensor_c},
      })
      .eval();
  py::buffer_info out_info = copy_op.getInfo(tensor_c);
  tensor_a.free();
  tensor_b.free();
  tensor_c.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);

  py::object base = py::cast(out_info.ptr);
  return py::array_t<float>(out_info, base);
}

PYBIND11_MODULE(_pyuTensor, m) {
  m.doc() = "pybind11 uTensor plugin";  // optional module docstring
  m.def("matmul", &matmul, "matmul", py::arg("a"), py::arg("b"));
}
