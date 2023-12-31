#include "conv.hpp"

#include "fast_copyop.hpp"
#include "uTensor/ops/Convolution.hpp"

using uTensor::Context;
using uTensor::RamTensor;
using uTensor::Tensor;
using uTensor::ReferenceOperators::Conv2dOperator;

static uTensor::localCircularArenaAllocator<1024> meta_allocator;
static uTensor::localCircularArenaAllocator<1024> ram_allocator;

py::array_t<float> conv2d_f(const py::array_t<float> &input,
                            const py::array_t<float> &filter,
                            const py::array_t<float> &bias,
                            std::array<uint16_t, 4> strides,
                            std::string padding) {
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  py::buffer_info info_input = input.request(), info_filter = filter.request(),
                  info_bias = bias.request();
  if (info_input.ndim != 4 || info_filter.ndim != 4) {
    throw py::value_error("input and filter should be both 4-dims array");
  }
  if (info_input.shape[3] != info_filter.shape[2]) {
    throw py::value_error(
        "in-channels must be the same for the input and filter");
  }
  if (info_filter.shape[3] != info_bias.shape[0]) {
    throw py::value_error(
        "the number of bias is not the same as filter out-channels");
  }
  uTensor::Padding padding_;
  if (padding == "VALID") {
    padding_ = uTensor::VALID;
  } else if (padding == "SAME") {
    padding_ = uTensor::SAME;
  } else {
    padding_ = uTensor::UNKNOWN;
  }

  CopyOperator copy_op;
  Conv2dOperator<float> conv_op(strides, padding_);
  Tensor tensor_input = new RamTensor(
      {
          static_cast<uint16_t>(info_input.shape[0]),
          static_cast<uint16_t>(info_input.shape[1]),
          static_cast<uint16_t>(info_input.shape[2]),
          static_cast<uint16_t>(info_input.shape[3]),

      },
      flt);
  Tensor tensor_filter = new RamTensor(
      {
          static_cast<uint16_t>(info_filter.shape[0]),
          static_cast<uint16_t>(info_filter.shape[1]),
          static_cast<uint16_t>(info_filter.shape[2]),
          static_cast<uint16_t>(info_filter.shape[3]),
      },
      flt);
  Tensor tensor_bias =
      new RamTensor({static_cast<uint16_t>(info_bias.shape[0])}, flt);
  Tensor tensor_out =
      new RamTensor({static_cast<uint16_t>(info_input.shape[0]), 1}, flt);
  conv_op
      .set_inputs({{Conv2dOperator<float>::in, tensor_input},
                   {Conv2dOperator<float>::filter, tensor_filter},
                   {Conv2dOperator<float>::bias, tensor_bias}})
      .set_outputs({{Conv2dOperator<float>::out, tensor_out}});
  py::buffer_info info = copy_op.getInfo(tensor_out);
  tensor_input.free();
  tensor_filter.free();
  tensor_bias.free();
  tensor_out.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);
  return py::array_t<float>(info);
}