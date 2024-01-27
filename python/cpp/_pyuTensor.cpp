#include <cstddef>

#include "allocator.hpp"
#include "conv.hpp"
#include "matmul.hpp"

PYBIND11_MODULE(_pyuTensor, m) {
  m.doc() = "pybind11 uTensor plugin";  // optional module docstring
  m.def("set_ram_total", &uTensor::python::set_ram_total, py::arg("capacity"));
  m.def("set_meta_total", &uTensor::python::set_meta_total,
        py::arg("capacity"));
  m.def("matmul", &matmul, "matmul", py::arg("a"), py::arg("b"));
  m.def("conv2d_f", &conv2d_f, "conv2d_f", py::arg("input"), py::arg("filter"),
        py::arg("bias"),
        py::arg("strides") = std::array<uint16_t, 4>({1, 1, 1, 1}),
        py::arg("padding") = "VALID");
}
