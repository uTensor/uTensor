#pragma once

#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "string"
#include "uTensor.h"

namespace py = pybind11;

py::array_t<float>
conv2d_f(const py::array_t<float, py::array::c_style> &input,
         const py::array_t<float, py::array::c_style> &filter,
         const py::array_t<float, py::array::c_style> &bias,
         std::array<uint16_t, 4> strides = {1, 1, 1, 1},
         std::string padding = "VALID");

template <typename T>
py::array_t<T> max_pool(py::array_t<T> input, std::array<uint16_t, 2> k_size,
                        std::array<uint16_t, 4> strides,
                        std::string padding = "VALID");
