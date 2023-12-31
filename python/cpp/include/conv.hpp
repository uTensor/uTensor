#pragma once

#include "pybind11/numpy.h"
#include "string"
#include "uTensor.h"

namespace py = pybind11;

void conv2d_f(const py::array_t<float> &input, const py::array_t<float> &filter,
              const py::array_t<float> &bias, std::string padding = "VALID");