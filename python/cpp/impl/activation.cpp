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

template <typename T, ttype uT>
py::array_t<T> relu_(const py::array_t<T> &input) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_input = input.request();
  CopyOperator copy_op;
  uTensor::ReferenceOperators::ReLUOperator<T> relu_op;
  Tensor tensor_input =
      new RamTensor({static_cast<uint16_t>(info_input.shape[0]),
                     static_cast<uint16_t>(info_input.shape[1])},
                    uT);
  Tensor tensor_output =
      new RamTensor({static_cast<uint16_t>(info_input.shape[0]),
                     static_cast<uint16_t>(info_input.shape[1])},
                    uT);
  copy_op.toTensor(info_input.ptr, tensor_input);
  relu_op
      .set_inputs(
          {{uTensor::ReferenceOperators::ReLUOperator<T>::in, tensor_input}})
      .set_outputs(
          {{uTensor::ReferenceOperators::ReLUOperator<T>::out, tensor_output}})
      .eval();
  py::buffer_info out_info = copy_op.getInfo(tensor_output);
  tensor_input.free();
  tensor_output.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);

  py::object base = py::cast(out_info.ptr);
  return py::array_t<T>(out_info, base);
}

py::array_t<float> relu_f(const py::array_t<float> &input) {
  return relu_<float, ttype::flt>(input);
}
py::array_t<int32_t> relu_i32(const py::array_t<int32_t> &input) {
  return relu_<int32_t, ttype::i32>(input);
}