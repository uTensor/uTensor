#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

template <typename T>
py::array_t<T> relu_q(const py::array_t<T> &input, float scale,
                      int32_t zero_point);
py::array_t<float> relu_f(const py::array_t<float> &input);