#pragma once

#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "string"
#include "uTensor.h"

namespace py = pybind11;

py::array_t<float> conv2d_f(const py::array_t<float> &input,
                            const py::array_t<float> &filter,
                            const py::array_t<float> &bias,
                            std::array<uint16_t, 4> strides = {1, 1, 1, 1},
                            std::string padding = "VALID");