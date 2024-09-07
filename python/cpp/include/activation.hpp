#pragma once

#include "pybind11/numpy.h"

namespace py = pybind11;

template <typename T> py::array_t<T> relu(const py::array_t<T> &input);
