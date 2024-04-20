#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

py::array_t<float> add_kernel(const py::array_t<float> &a,
                          const py::array_t<float> &b);

py::array_t<float> mul_kernel(const py::array_t<float> &a,
                            const py::array_t<float> &b);