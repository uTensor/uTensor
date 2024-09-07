#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

template <typename T>
py::array_t<T> add_kernel(const py::array_t<T> &a, const py::array_t<T> &b);

py::array_t<float> mul_kernel(const py::array_t<float> &a,
                              const py::array_t<float> &b);