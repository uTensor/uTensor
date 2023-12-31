#pragma once

#include <initializer_list>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "string"
#include "uTensor.h"

namespace py = pybind11;

void conv2d_f(const py::array_t<float> &input, const py::array_t<float> &filter,
              const py::array_t<float> &bias,
              std::initializer_list<uint16_t> strides = {1, 1, 1, 1},
              std::string padding = "VALID");