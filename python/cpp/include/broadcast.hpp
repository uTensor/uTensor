#pragma once

#include "uTensor/ops/Broadcast.hpp"

#include "pybind11/pybind11.h"


namespace py = pybind11;

class PyBroadcaster {
public:
    PyBroadcaster( const py::tuple &shape_a, const py::tuple &shape_b );
    py::tuple get_linear_idx( int idx_c ) const;
    py::tuple get_shape_c() const;
private:
    Broadcaster _bc;
};
