#ifndef UTENSOR_S_QUANTIZED_DWS_OPS_H
#define UTENSOR_S_QUANTIZED_DWS_OPS_H
#include "Convolution.hpp"
#include "context.hpp"
#include "depthwise_separable_convolution_kernels.hpp"
#include "operatorBase.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
DECLARE_ERROR(qDwsConvPerChannelMismatchError);

namespace TFLM {

// Keep this outside the Operator since we can include this file in the
// Optimized Ops and get option data.
static const int kMaxChannels = 32;  // was 256
constexpr int kDepthwiseConvQuantizedDimension = 3;

struct DWSConvOpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  // TODO(b/141139247): Allocate these dynamically when possible.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};
}  // namespace TFLM

template <typename Tout>
class QuantizedDepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

 public:
  QuantizedDepthwiseSeparableConvOperator();
  QuantizedDepthwiseSeparableConvOperator(
      std::initializer_list<uint16_t> strides, Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1});

  // activation basically only used for TESTING, USE AT YOUR OWN RISK
  QuantizedDepthwiseSeparableConvOperator(
      std::initializer_list<uint16_t> strides, Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1},
      const TFLM::TfLiteFusedActivation activation = TFLM::kTfLiteActNone);

  static void calculateOpData(
      const Tensor& input, const Tensor& filter, const Tensor& bias,
      Tensor& output, const uint16_t (&strides)[4], const Padding padding,
      const int depth_multiplier, const uint16_t (&dialations)[2],
      int output_shift, int32_t* per_channel_output_multiplier,
      int32_t* per_channel_output_shift, int& padding_height,
      int& padding_width, int32_t& output_multiplier,
      int32_t& output_activation_min, int32_t& output_activation_max,
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
  // DWSConvOpData (check the previous commit)
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
QuantizedDepthwiseSeparableConvOperator<
    Tout>::QuantizedDepthwiseSeparableConvOperator()
    : _stride{1, 1},
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
QuantizedDepthwiseSeparableConvOperator<Tout>::
    QuantizedDepthwiseSeparableConvOperator(
        std::initializer_list<uint16_t> strides, Padding padding,
        const int depth_multiplier, const uint16_t (&dialation)[2],
        TFLM::TfLiteFusedActivation activation)
    : _padding(padding),
      depth_multiplier(depth_multiplier),
      _dialation{dialation[0], dialation[1]},
      activation(activation) {
  int i = 0;
  for (auto s : strides) {
    _stride[i++] = s;
  }
}

template <typename Tout>
void QuantizedDepthwiseSeparableConvOperator<Tout>::calculateOpData(
    const Tensor& input, const Tensor& filter, const Tensor& bias,
    Tensor& output, const uint16_t (&strides)[4], const Padding padding,
    const int depth_multiplier, const uint16_t (&dialations)[2],
    int output_shift, int32_t* per_channel_output_multiplier,
    int32_t* per_channel_output_shift, int32_t& padding_height,
    int32_t& padding_width, int32_t& output_multiplier,
    int32_t& output_activation_min, int32_t& output_activation_max,
    TFLM::TfLiteFusedActivation activation) {
  const int channels_out = filter->get_shape()[3];
  const int width = input->get_shape()[2];
  const int height = input->get_shape()[1];
  const int filter_width = filter->get_shape()[2];
  const int filter_height = filter->get_shape()[1];
  const int stride_height = strides[0];
  const int stride_width = strides[1];

  int unused_output_height, unused_output_width;

  // Luckily our padding enum matches up so we can just cast this shit
  TFLM::ComputePaddingHeightWidth(stride_height, stride_width, 1, 1, height,
                                  width, filter_height, filter_width,
                                  &padding_height, &padding_width,
                                  static_cast<TFLM::TfLitePadding>(padding),
                                  &unused_output_height, &unused_output_width);

  int num_channels =
      filter->get_shape()[TFLM::kDepthwiseConvQuantizedDimension];
  QuantizationParams affine_quantization = filter->get_quantization_params();
  const bool is_per_channel = affine_quantization.num_channels() > 1;

  if (is_per_channel) {
    //  Currently only Int8 is supported for per channel quantization.
    // TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    // TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);
    if (!(affine_quantization.num_channels() == num_channels)) {
      Context::get_default_context()->throwError(
          new qDwsConvPerChannelMismatchError);
    }
    if (!(num_channels ==
          filter->get_shape()
              [affine_quantization
                   .num_channels()])) {  // FIXME:
                                         // affine_quantization.num_channels()-1?
      Context::get_default_context()->throwError(
          new qDwsConvPerChannelMismatchError);
    }
  }

  const float input_scale =
      input->get_quantization_params().get_scale_for_channel(0);
  // const float output_scale = output->params.scale;
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
void QuantizedDepthwiseSeparableConvOperator<Tout>::compute() {
  AllocatorInterface* ram_allocator =
      Context::get_default_context()->get_ram_data_allocator();
  const TensorShape& in_shape = inputs[in].tensor()->get_shape();
  const TensorShape& df_shape = inputs[filter].tensor()->get_shape();
  const TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
  const TensorShape& out_shape = outputs[out].tensor()->get_shape();

  if (in_shape[3] != df_shape[2]) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (bias_shape[0] != 1 || bias_shape[1] != 1) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }

  TFLM::DepthwiseParams op_params;
  TFLM::TfLitePaddingValues paddingVals;

  int num_channels = inputs[filter]
                         .tensor()
                         ->get_shape()[TFLM::kDepthwiseConvQuantizedDimension];
  per_channel_output_multiplier = reinterpret_cast<int32_t*>(
      ram_allocator->allocate(sizeof(int32_t) * num_channels));
  per_channel_output_shift = reinterpret_cast<int32_t*>(
      ram_allocator->allocate(sizeof(int32_t) * num_channels));

  // Bind these params to a Handle so they dont accidentally get thrown away on
  // possible rebalance
  Handle per_channel_output_multiplier_h(per_channel_output_multiplier);
  Handle per_channel_output_shift_h(per_channel_output_shift);
  ram_allocator->bind(per_channel_output_multiplier,
                      &per_channel_output_multiplier_h);
  ram_allocator->bind(per_channel_output_shift, &per_channel_output_shift_h);

  calculateOpData(inputs[in].tensor(), inputs[filter].tensor(),
                  inputs[bias].tensor(), outputs[out].tensor(), _stride,
                  _padding, depth_multiplier, _dialation, output_shift,
                  per_channel_output_multiplier, per_channel_output_shift,
                  paddingVals.width, paddingVals.height, output_multiplier,
                  output_activation_min, output_activation_max,
                  activation  // Basically only used for test
  );

  op_params.padding_type = TFLM::PaddingType::kSame;
  // op_params.padding_type = static_cast<TFLM::PaddingType>(_padding);
  op_params.padding_values.width = paddingVals.width;
  op_params.padding_values.height = paddingVals.height;
  op_params.stride_width = _stride[1];
  op_params.stride_height = _stride[0];
  op_params.dilation_width_factor = _dialation[1];
  op_params.dilation_height_factor = _dialation[0];
  op_params.depth_multiplier = depth_multiplier;
  op_params.input_offset =
      -inputs[in].tensor()->get_quantization_params().get_zeroP_for_channel(0);
  ;
  op_params.weights_offset = 0;
  op_params.output_offset =
      outputs[out].tensor()->get_quantization_params().get_zeroP_for_channel(
          0);  // output->params.zero_point;
  // TODO(b/130439627): Use calculated value for clamping.
  op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

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

}  // namespace uTensor
#endif
