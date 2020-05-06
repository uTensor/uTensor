#ifndef UTENSOR_CMSIS_CONV_H
#define UTENSOR_CMSIS_CONV_H

#include <algorithm>
#include <limits>

#include "Convolution_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {

template <typename T>
class CmsisConvOperator : public OperatorInterface<3, 1>, FastOperator {
 public:
  enum names_in : uint8_t { input, filter, bias };
  enum names_out : uint8_t { output };
  CmsisConvOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding), 
        output_activation_min(std::numeric_limits<int8_t>::min(),
        output_activation_max(std::numeric_limits<int8_t>::max() {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    const TensorShape& filter_shape = inputs[filter].tensor()->get_shape();
    const TensorShape& input_shape = inputs[input].tensor()->get_shape();
    const TensorShape& output_shape = outputs[output].tensor()->get_shape();
    ConvFilter<T> conv(*inputs[filter].tensor);
    generic_convolution_kernel<T, ConvFilter<T>>(
        outputs[out].tensor(), inputs[in].tensor(), conv, _padding, _stride);

    const int32_t* bias = nullptr;
    const int8_t* kernel = nullptr;
    const int8_t* input = nullptr;
    int8_t* output = nullptr;

    // Wrap in an if somehow
    const TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
    size_t bias_read = inputs[bias].tensor().get_readable_block(
        bias, bias_shape.get_linear_size(), 0);  // Attempt to read all of bias
    size_t kernel_read = inputs[filter].tensor().get_readable_block(
        kernel, filter_shape.get_linear_size(),
        0);  // Attempt to read all of kernel
    size_t input_read = inputs[input].tensor().get_readable_block(
        input, input_shape.get_linear_size(),
        0);  // Attempt to read all of input
    size_t output_read = outputs[output].tensor().get_readable_block(
        output, output_shape.get_linear_size(),
        0);  // Attempt to read all of input
    // TODO replace the direct shape access with constants that are more clear
    q15_t* buffer_a =
        Context::get_default_context()->get_metadata_allocator()->allocate(
            arm_convolve_s8_get_buffer_size(input_shape[3], filter_shape[2],
                                            filter_shape[1]));

    arm_status arm_convolve_s8(
        input, input_shape[2], input_shape[1], input_shape[3], input_shape[0],
        kernel, filter_shape[0], filter_shape[2], filter_shape[1],
        const uint16_t pad_x, const uint16_t pad_y, _stride[1], _stride[0],
        bias, output, const int32_t* output_shift, const int32_t* output_mult,
        const int32_t out_offset, const int32_t input_offset,
        const int32_t output_activation_min,
        const int32_t output_activation_max, output_shape[1], output_shape[0],
        buffer_a);
    Context::get_default_context()->get_metadata_allocator()->deallocate(
        buffer_a);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
  // Do something with this eventually
  const int32_t output_activation_min;
  const int32_t output_activation_max;
};

template <typename T>
class CmsisDepthwiseSeparableConvOperator : public OperatorInterface<3, 1>,
                                            FastOperator {
 public:
  enum names_in : uint8_t { in, depthwise_filter, pointwise_filter };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  CmsisDepthwiseSeparableConvOperator(std::initializer_list<uint16_t> strides,
                                 Padding padding)
      : _padding(padding),
        output_activation_min(std::numeric_limits<int8_t>::min(),
        output_activation_max(std::numeric_limits<int8_t>::max() {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape = (*inputs[in].tensor)->get_shape();
    TensorShape& df_shape = (*inputs[depthwise_filter].tensor)->get_shape();
    TensorShape& pf_shape = (*inputs[pointwise_filter].tensor)->get_shape();
    TensorShape& out_shape = (*outputs[out].tensor)->get_shape();

    if (in_shape[3] != df_shape[2]) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    if (pf_shape[0] != 1 || pf_shape[1] != 1) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    depthwise_separable_convolution_kernel<T>(
        *outputs[out].tensor, *inputs[in].tensor,
        *inputs[depthwise_filter].tensor, *inputs[pointwise_filter].tensor,
        _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
  // Do something with this eventually
  const int32_t output_activation_min;
  const int32_t output_activation_max;
};

}  // namespace uTensor
#endif
