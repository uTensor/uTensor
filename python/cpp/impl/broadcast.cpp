#include "broadcast.hpp"

#include "uTensor/core/types.hpp"

PyBroadcaster::PyBroadcaster(const py::tuple &shape_a,
                             const py::tuple &shape_b) {
  TensorShape ts_a(0), ts_b(0);
  for (int i = 0; i < shape_a.size(); i++)
    ts_a[i] = static_cast<uint16_t>(shape_a[i].cast<int>());
  for (int i = 0; i < shape_b.size(); i++)
    ts_b[i] = static_cast<uint16_t>(shape_b[i].cast<int>());
  ts_a.update_dims();
  ts_b.update_dims();
  try {
    _bc.set_shape(ts_a, ts_b);
  } catch (const std::runtime_error &e) {
    throw py::value_error(e.what());
  }
}

py::tuple PyBroadcaster::get_linear_idx(int idx_c) const {
  std::pair<uint32_t, uint32_t> indices;
  try {
    indices = _bc.get_linear_idx(idx_c);
  } catch (const std::runtime_error &e) {
    throw py::value_error(e.what());
  }
  return py::make_tuple(indices.first, indices.second);
}

py::tuple PyBroadcaster::get_output_shape() const {
  TensorShape ts = _bc.get_output_shape();
  py::tuple shape(ts.num_dims());
  for (int i = 0; i < ts.num_dims(); i++) shape[i] = ts[i];
  return shape;
}