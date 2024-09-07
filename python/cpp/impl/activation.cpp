#include "activation.hpp"

#include "allocator.hpp"
#include "fast_copyop.hpp"
#include "uTensor/core/types.hpp"
#include "uTensor/ops/ActivationFncs.hpp"

namespace py = pybind11;
using uTensor::Context;
using uTensor::RamTensor;
using uTensor::Tensor;
using uTensor::python::get_meta_allocator;
using uTensor::python::get_ram_allocator;

template <typename T> py::array_t<T> relu(const py::array_t<T> &input) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_input = input.request();
  CopyOperator copy_op;
  uTensor::ReferenceOperators::ReLUOperator<T> relu_op;
  TensorShape in_shape(0);

  for (int idx = 0; idx < info_input.ndim; idx++) {
    in_shape[idx] = info_input.shape[idx];
  }
  in_shape.update_dims();
  auto elem_type = ttype_from<T>::type;
  Tensor tensor_input = new RamTensor(in_shape, elem_type);
  Tensor tensor_output = new RamTensor(in_shape, elem_type);
  copy_op.toTensor(info_input.ptr, tensor_input);
  relu_op
      .set_inputs({{uTensor::ReferenceOperators::ReLUOperator<float>::in,
                    tensor_input}})
      .set_outputs({{uTensor::ReferenceOperators::ReLUOperator<float>::out,
                     tensor_output}})
      .eval();
  py::buffer_info out_info = copy_op.getInfo(tensor_output);
  tensor_input.free();
  tensor_output.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);

  py::object base = py::cast(out_info.ptr);
  return py::array_t<T>(out_info, base);
}

template py::array_t<float> relu(const py::array_t<float> &input);
template py::array_t<int8_t> relu(const py::array_t<int8_t> &input);
template py::array_t<int16_t> relu(const py::array_t<int16_t> &input);
template py::array_t<int32_t> relu(const py::array_t<int32_t> &input);
