#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

py::array_t<float> relu_f(const py::array_t<float> &input);