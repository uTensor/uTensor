#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

py::array_t<float> matmul(const py::array_t<float> &a,
                          const py::array_t<float> &b);