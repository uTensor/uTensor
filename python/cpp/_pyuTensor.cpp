#include <cstddef>

#include "activation.hpp"
#include "allocator.hpp"
#include "arithmetic_kernels.hpp"
#include "broadcast.hpp"
#include "conv.hpp"
#include "matmul.hpp"

PYBIND11_MODULE(_pyuTensor, m) {
  m.doc() = "pybind11 uTensor plugin"; // optional module docstring
  m.def("set_ram_total", &uTensor::python::set_ram_total, py::arg("capacity"));
  m.def("set_meta_total", &uTensor::python::set_meta_total,
        py::arg("capacity"));
  m.def("matmul", &matmul, "matmul", py::arg("a"), py::arg("b"));
  m.def("conv2d_f", &conv2d_f, "conv2d_f", py::arg("input"), py::arg("filter"),
        py::arg("bias"),
        py::arg("strides") = std::array<uint16_t, 4>({1, 1, 1, 1}),
        py::arg("padding") = "VALID");
  m.def("max_pool_f", &(max_pool<float>), "max_pool_f", py::arg("input"),
        py::arg("k_size"), py::arg("strides"), py::arg("padding") = "VALID");
  m.def("max_pool_i8", &(max_pool<int8_t>), "max_pool_i8", py::arg("input"),
        py::arg("k_size"), py::arg("strides"), py::arg("padding") = "VALID");
  m.def("max_pool_u8", &(max_pool<uint8_t>), "max_pool_u8", py::arg("input"),
        py::arg("k_size"), py::arg("strides"), py::arg("padding") = "VALID");
  m.def("add_kernel_f", &add_kernel<float>, "add_kernel_f", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_i8", &add_kernel<int8_t>, "add_kernel_i8", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_u8", &add_kernel<uint8_t>, "add_kernel_u8", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_i16", &add_kernel<int16_t>, "add_kernel_i16", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_u16", &add_kernel<uint16_t>, "add_kernel_u16", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_i32", &add_kernel<int32_t>, "add_kernel_i32", py::arg("a"),
        py::arg("b"));
  m.def("add_kernel_u32", &add_kernel<uint32_t>, "add_kernel_u32", py::arg("a"),
        py::arg("b"));
  m.def("mul_kernel", &mul_kernel, "mul_kernel", py::arg("a"), py::arg("b"));
  py::class_<PyBroadcaster>(m, "Broadcaster")
      .def(py::init<const py::tuple &, const py::tuple &>())
      .def("get_output_shape", &PyBroadcaster::get_output_shape)
      .def("get_linear_idx", &PyBroadcaster::get_linear_idx);
  m.def("relu_f", &relu_f, "relu_f", py::arg("input"));
  m.def("relu_i8", &relu_q<int8_t>, "relu_i8", py::arg("input"),
        py::arg("scale"), py::arg("zero_point"));
  m.def("relu_i16", &relu_q<int16_t>, "relu_i16", py::arg("input"),
        py::arg("scale"), py::arg("zero_point"));
  m.def("relu_i32", &relu_q<int32_t>, "relu_i32", py::arg("input"),
        py::arg("scale"), py::arg("zero_point"));
}
