#ifndef UTENSOR_S_QUANTIZED_DWS_OPS_H
#define UTENSOR_S_QUANTIZED_DWS_OPS_H
#include "Convolution.hpp"
#include "context.hpp"
#include "depthwise_separable_convolution_kernels.hpp"
#include "operatorBase.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
DECLARE_ERROR(qDwsConvPerChannelMismatchError);

namespace TflmSymQuantOps {

// Keep this outside the Operator since we can include this file in the
// Optimized Ops and get option data.
constexpr int kDepthwiseConvQuantizedDimension = 3;

template <typename Tout>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

 public:
  DepthwiseSeparableConvOperator();
  /*
  DepthwiseSeparableConvOperator(
      const uint16_t (&strides)[2], Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1});
  DepthwiseSeparableConvOperator(
      const uint16_t (&strides)[4], Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1});
  */
  // activation basically only used for TESTING, USE AT YOUR OWN RISK
  DepthwiseSeparableConvOperator(
      const uint16_t (&strides)[2], Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1},
      const TFLM::TfLiteFusedActivation activation = TFLM::kTfLiteActNone);
  //// activation basically only used for TESTING, USE AT YOUR OWN RISK
  // DepthwiseSeparableConvOperator(
  //    std::initializer_list<uint16_t> strides, Padding padding,
  //    const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1},
  //    const TFLM::TfLiteFusedActivation activation = TFLM::kTfLiteActNone);

  static void calculateOpData(
      const Tensor& input, const Tensor& filter, const Tensor& bias,
      Tensor& output, const uint16_t (&strides)[4], const Padding padding,
      const uint16_t (&dialations)[2], int output_shift,
      int32_t* per_channel_output_multiplier, int32_t* per_channel_output_shift,
      int32_t& padding_height, int32_t& padding_width,
      int32_t& output_multiplier, int32_t& output_activation_min,
      int32_t& output_activation_max,
      TFLM::TfLiteFusedActivation =
          TFLM::kTfLiteActNone  // Make this param basically not required
  );

 protected:
  virtual void compute();

 private:
  // TfLiteDepthwiseConvParams
  // Set by constructors
  uint16_t _stride[4];
  Padding _padding;
  int depth_multiplier;
  uint16_t _dialation[2];

  int32_t output_multiplier;
  int output_shift;
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // BS
  TFLM::TfLiteFusedActivation activation;
};

template <typename Tout>
DepthwiseSeparableConvOperator<
    Tout>::DepthwiseSeparableConvOperator()
    : _stride{1, 1, 1, 1},
      _padding(SAME),
      depth_multiplier(1),
      _dialation{1, 1},
      output_multiplier(1),
      output_shift(0),
      per_channel_output_multiplier(nullptr),
      per_channel_output_shift(nullptr),
      output_activation_min(std::numeric_limits<Tout>::min()),
      output_activation_max(std::numeric_limits<Tout>::max()) {}

template <typename Tout>
DepthwiseSeparableConvOperator<Tout>::
    DepthwiseSeparableConvOperator(
        const uint16_t (&strides)[2], Padding padding,
        const int depth_multiplier, const uint16_t (&dialation)[2],
        TFLM::TfLiteFusedActivation activation)
    : _stride{1, strides[0], strides[1], 1},
      _padding(padding),
      depth_multiplier(depth_multiplier),
      _dialation{dialation[0], dialation[1]},
      activation(activation) {}

template <typename Tout>
void DepthwiseSeparableConvOperator<Tout>::calculateOpData(
    const Tensor& input, const Tensor& filter, const Tensor& bias,
    Tensor& output, const uint16_t (&strides)[4], const Padding padding,
    const uint16_t (&dialations)[2], int output_shift,
    int32_t* per_channel_output_multiplier, int32_t* per_channel_output_shift,
    int32_t& padding_height, int32_t& padding_width, int32_t& output_multiplier,
    int32_t& output_activation_min, int32_t& output_activation_max,
    TFLM::TfLiteFusedActivation activation) {
  const int channels_out = filter->get_shape()[3];
  const int width = input->get_shape()[2];
  const int height = input->get_shape()[1];
  const int filter_width = filter->get_shape()[2];
  const int filter_height = filter->get_shape()[1];
  const int stride_height = strides[1];
  const int stride_width = strides[2];

  int unused_output_height, unused_output_width;

  // Luckily our padding enum matches up so we can just cast this shit
  TFLM::ComputePaddingHeightWidth(stride_height, stride_width, 1, 1, height,
                                  width, filter_height, filter_width,
                                  &padding_height, &padding_width,
                                  static_cast<TFLM::TfLitePadding>(padding),
                                  &unused_output_height, &unused_output_width);

  int num_channels =
      filter->get_shape()[kDepthwiseConvQuantizedDimension];
  QuantizationParams affine_quantization = filter->get_quantization_params();
  const bool is_per_channel = affine_quantization.num_channels() > 1;
  // dws conv should be per channel quantized
  if (!is_per_channel) {
    Context::get_default_context()->throwError(
        new InvalidQuantizationSchemeError);
  }
  // check the types are correctly following the spec
  if (input->get_type() != i8 || filter->get_type() != i8 ||
      bias->get_type() != i32) {
    Context::get_default_context()->throwError(new InvalidTensorDataTypeError);
  }
  if (!(affine_quantization.num_channels() == num_channels)) {
    Context::get_default_context()->throwError(
        new qDwsConvPerChannelMismatchError);
  }

  // PopulateConvolutionQuantizationParams
  // https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/kernels/kernel_util.cc#L42
  const float input_scale =
      input->get_quantization_params().get_scale_for_channel(0);
  const float output_scale =
      output->get_quantization_params().get_scale_for_channel(0);

  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along
    // the quantization dimension (channels_out).
    const float scale =
        is_per_channel
            ? filter->get_quantization_params().get_scale_for_channel(i)
            : filter->get_quantization_params().get_scale_for_channel(0);
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    TFLM::QuantizeMultiplier(effective_output_scale, &significand,
                             &channel_shift);
    reinterpret_cast<int32_t*>(per_channel_output_multiplier)[i] = significand;
    reinterpret_cast<int32_t*>(per_channel_output_shift)[i] = channel_shift;

    // if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    TFLM::CalculateActivationRangeQuantized(
        activation, output, &output_activation_min, &output_activation_max);
  }
}

template <typename Tout>
void DepthwiseSeparableConvOperator<Tout>::compute() {
  AllocatorInterface* ram_allocator =
      Context::get_default_context()->get_ram_data_allocator();
  const TensorShape& in_shape = inputs[in].tensor()->get_shape();
  const TensorShape& df_shape = inputs[filter].tensor()->get_shape();
  const TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
  const TensorShape& out_shape = outputs[out].tensor()->get_shape();

  /* These are not correct
  if (in_shape[3] != df_shape[2]) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (bias_shape[0] != 1 || bias_shape[1] != 1) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  */

  TFLM::DepthwiseParams op_params;
  // This silly shit is just a width and height
  TFLM::TfLitePaddingValues paddingVals;

  int num_channels = inputs[filter]
                         .tensor()
                         ->get_shape()[kDepthwiseConvQuantizedDimension];
  // Bind these params to a Handle so they dont accidentally get thrown away on
  // possible rebalance
  per_channel_output_multiplier = reinterpret_cast<int32_t*>(
      ram_allocator->allocate(sizeof(int32_t) * num_channels));
  Handle per_channel_output_multiplier_h(per_channel_output_multiplier);
  ram_allocator->bind(per_channel_output_multiplier,
                      &per_channel_output_multiplier_h);
  per_channel_output_shift = reinterpret_cast<int32_t*>(
      ram_allocator->allocate(sizeof(int32_t) * num_channels));
  Handle per_channel_output_shift_h(per_channel_output_shift);
  ram_allocator->bind(per_channel_output_shift, &per_channel_output_shift_h);

  calculateOpData(inputs[in].tensor(), inputs[filter].tensor(),
                  inputs[bias].tensor(), outputs[out].tensor(), _stride,
                  _padding, _dialation, output_shift,
                  per_channel_output_multiplier, per_channel_output_shift,
                  paddingVals.height, paddingVals.width, output_multiplier,
                  output_activation_min, output_activation_max,
                  activation  // Basically only used for test
  );

  // op_params.padding_type = TFLM::PaddingType::kSame;
  op_params.padding_type = static_cast<TFLM::PaddingType>(_padding);
  op_params.padding_values.width = paddingVals.width;
  op_params.padding_values.height = paddingVals.height;
  op_params.stride_width = _stride[1];
  op_params.stride_height = _stride[2];
  op_params.dilation_width_factor = _dialation[1];
  op_params.dilation_height_factor = _dialation[0];
  op_params.depth_multiplier = depth_multiplier;
  op_params.input_offset =
      -inputs[in].tensor()->get_quantization_params().get_zeroP_for_channel(0);
  ;
  op_params.weights_offset = 0;
  op_params.output_offset =
      outputs[out].tensor()->get_quantization_params().get_zeroP_for_channel(0);

  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  TFLM::DepthwiseConvPerChannel<Tout>(
      op_params, per_channel_output_multiplier, per_channel_output_shift,
      inputs[in].tensor(), inputs[filter].tensor(), inputs[bias].tensor(),
      outputs[out].tensor());

  // Free up any allocated bits
  ram_allocator->unbind(per_channel_output_shift, &per_channel_output_shift_h);
  ram_allocator->unbind(per_channel_output_multiplier,
                        &per_channel_output_multiplier_h);
  ram_allocator->deallocate(per_channel_output_shift);
  ram_allocator->deallocate(per_channel_output_multiplier);
}

}
}  // namespace uTensor
#endif
