#include "conv.hpp"

#include <cmath>

#include "allocator.hpp"
#include "fast_copyop.hpp"
#include "uTensor/ops/Convolution.hpp"

using uTensor::Context;
using uTensor::RamTensor;
using uTensor::Tensor;
using uTensor::python::get_meta_allocator;
using uTensor::python::get_ram_allocator;
using uTensor::ReferenceOperators::Conv2dOperator;
using namespace uTensor::ReferenceOperators::Conv2dConstants;

static void _compute_output_shape(py::buffer_info &info_input,
                                  uTensor::Padding padding,
                                  uint16_t kernel_height, uint16_t kernel_width,
                                  uint16_t stride_height, uint16_t stride_width,
                                  uint16_t &out_height, uint16_t &out_width) {
  switch (padding) {
  case uTensor::VALID:
    out_height = static_cast<uint16_t>(std::ceil(
        static_cast<float>((info_input.shape[1] - kernel_height + 1)) /
        static_cast<float>(stride_height)));
    out_width = static_cast<uint16_t>(
        std::ceil(static_cast<float>(info_input.shape[2] - kernel_width + 1) /
                  static_cast<float>(stride_width)));
    break;
  case uTensor::SAME:
    out_height = static_cast<uint16_t>(
        std::ceil(static_cast<float>(info_input.shape[1]) /
                  static_cast<float>(stride_height)));
    out_width = static_cast<uint16_t>(
        std::ceil(static_cast<float>(info_input.shape[2]) /
                  static_cast<float>(stride_width)));
    break;
  case uTensor::UNKNOWN:
    throw py::value_error("invalid padding value, support only SAME and VALID");
  }
}

py::array_t<float>
conv2d_f(const py::array_t<float, py::array::c_style> &input,
         const py::array_t<float, py::array::c_style> &filter,
         const py::array_t<float, py::array::c_style> &bias,
         std::array<uint16_t, 4> strides, std::string padding) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_input = input.request(), info_filter = filter.request(),
                  info_bias = bias.request();
  if (info_input.ndim != 4 || info_filter.ndim != 4) {
    throw py::value_error("input and filter should be both 4-dims array");
  }
  if (info_input.shape[3] != info_filter.shape[filter_in_channels_dim]) {
    throw py::value_error(
        "in-channels must be the same for the input and filter");
  }
  if (info_filter.shape[filter_out_channels_dim] != info_bias.shape[0]) {
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
  // https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
  uint16_t out_height, out_width;
  _compute_output_shape(info_input, padding_,
                        info_filter.shape[filter_height_dim],
                        info_filter.shape[filter_width_dim], strides[1],
                        strides[2], out_height, out_width);

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
  copy_op.toTensor(info_input.ptr, tensor_input);
  copy_op.toTensor(info_filter.ptr, tensor_filter);
  copy_op.toTensor(info_bias.ptr, tensor_bias);
  Tensor tensor_out = new RamTensor(
      {static_cast<uint16_t>(info_input.shape[0]), out_height, out_width,
       static_cast<uint16_t>(info_filter.shape[filter_out_channels_dim])},
      flt);
  conv_op
      .set_inputs({{Conv2dOperator<float>::in, tensor_input},
                   {Conv2dOperator<float>::filter, tensor_filter},
                   {Conv2dOperator<float>::bias, tensor_bias}})
      .set_outputs({{Conv2dOperator<float>::out, tensor_out}})
      .eval();
  py::buffer_info info = copy_op.getInfo(tensor_out);
  tensor_input.free();
  tensor_filter.free();
  tensor_bias.free();
  tensor_out.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);
  return py::array_t<float>(info);
}

template <typename T>
py::array_t<T> max_pool(py::array_t<T> input, std::array<uint16_t, 2> k_size,
                        std::array<uint16_t, 4> strides, std::string padding) {
  Context::get_default_context()->set_ram_data_allocator(get_ram_allocator());
  Context::get_default_context()->set_metadata_allocator(get_meta_allocator());
  py::buffer_info info_input = input.request();
  uTensor::Padding padding_;
  if (padding == "VALID") {
    padding_ = uTensor::VALID;
  } else if (padding == "SAME") {
    padding_ = uTensor::SAME;
  } else {
    padding_ = uTensor::UNKNOWN;
  }
  CopyOperator copy_op;
  uTensor::ReferenceOperators::MaxPoolOperator<T> max_pool_op(k_size, strides,
                                                              padding_);
  auto elem_type = ttype_from<T>::type;
  uint16_t out_height, out_width;
  _compute_output_shape(info_input, padding_, k_size[0], k_size[1], strides[1],
                        strides[2], out_height, out_width);
  Tensor tensor_out = new RamTensor(
      {
          static_cast<uint16_t>(info_input.shape[0]),
          out_height,
          out_width,
          static_cast<uint16_t>(info_input.shape[3]),
      },
      elem_type);
  Tensor tensor_input = new RamTensor(
      {
          static_cast<uint16_t>(info_input.shape[0]),
          static_cast<uint16_t>(info_input.shape[1]),
          static_cast<uint16_t>(info_input.shape[2]),
          static_cast<uint16_t>(info_input.shape[3]),

      },
      elem_type);
  copy_op.toTensor(info_input.ptr, tensor_input);
  max_pool_op
      .set_inputs(
          {{uTensor::ReferenceOperators::MaxPoolOperator<T>::in, tensor_input}})
      .set_outputs(
          {{uTensor::ReferenceOperators::MaxPoolOperator<T>::out, tensor_out}})
      .eval();
  py::buffer_info info = copy_op.getInfo(tensor_out);
  tensor_input.free();
  tensor_out.free();
  Context::get_default_context()->set_ram_data_allocator(nullptr);
  Context::get_default_context()->set_metadata_allocator(nullptr);
  return py::array_t<T>(info);
}

template py::array_t<float> max_pool<float>(py::array_t<float> input,
                                            std::array<uint16_t, 2> k_size,
                                            std::array<uint16_t, 4> strides,
                                            std::string padding);

template py::array_t<int8_t> max_pool<int8_t>(py::array_t<int8_t> input,
                                              std::array<uint16_t, 2> k_size,
                                              std::array<uint16_t, 4> strides,
                                              std::string padding);

template py::array_t<uint8_t> max_pool<uint8_t>(py::array_t<uint8_t> input,
                                                std::array<uint16_t, 2> k_size,
                                                std::array<uint16_t, 4> strides,
                                                std::string padding);
