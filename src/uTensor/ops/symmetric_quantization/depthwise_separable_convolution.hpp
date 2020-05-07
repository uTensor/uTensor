#ifndef UTENSOR_S_QUANTIZED_DWS_OPS_H
#define UTENSOR_S_QUANTIZED_DWS_OPS_H
#include "context.hpp"
#include "operatorBase.hpp"
#include "depthwise_separable_convolution_kernels.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
DECLARE_ERROR(qDwsConvPerChannelMismatchError);

namespace TFLM {

// Keep this outside the Operator since we can include this file in the Optimized Ops and get option data.
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
} //TFLM

template <typename Tout>
class QuantizedDepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };
  

public:
  QuantizedDepthwiseSeparableConvOperator() = default;
  QuantizedDepthwiseSeparableConvOperator(TFLM::TfLiteDepthwiseConvParams& _params);

  QuantizedDepthwiseSeparableConvOperator& set_params(
      TFLM::TfLiteDepthwiseConvParams& _params);

  void calculateOpData(TFLM::DWSConvOpData* data);

  // FIXME: remove this method
  // QuantizedDepthwiseSeparableConvOperator& set_params(DepthwiseParams&& _params, const
  // int32_t&& _output_multiplier, const int32_t&& _output_shift) {
  //   //params = _params;
  //   output_multiplier = _output_multiplier;
  //   output_shift = _output_shift;
  //   return *this;
  // }

 protected:
  virtual void compute() {
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    TensorShape& df_shape = inputs[filter].tensor()->get_shape();
    TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
    TensorShape& out_shape = outputs[out].tensor()->get_shape();

    if (in_shape[3] != df_shape[2]) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    if (bias_shape[0] != 1 || bias_shape[1] != 1) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }

    // TODO: compute param here

    TFLM::DWSConvOpData data;
    calculateOpData(&data);

    TFLM::DepthwiseParams op_params;
    op_params.padding_type = TFLM::PaddingType::kSame;
    op_params.padding_values.width = data.padding.width;
    op_params.padding_values.height = data.padding.height;
    op_params.stride_width = params.stride_width;
    op_params.stride_height = params.stride_height;
    op_params.dilation_width_factor = params.dilation_width_factor;
    op_params.dilation_height_factor = params.dilation_height_factor;
    op_params.depth_multiplier = params.depth_multiplier;
    op_params.input_offset =
        -inputs[in].tensor()->get_quantization_params().get_zeroP_for_channel(
            0);
    ;
    op_params.weights_offset = 0;
    op_params.output_offset =
        outputs[out].tensor()->get_quantization_params().get_zeroP_for_channel(
            0);  // output->params.zero_point;
    // TODO(b/130439627): Use calculated value for clamping.
    op_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
    op_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

    TFLM::DepthwiseConvPerChannel<Tout>(op_params, data.per_channel_output_multiplier,
                                  data.per_channel_output_shift,
                                  inputs[in].tensor(), inputs[filter].tensor(),
                                  inputs[bias].tensor(), outputs[out].tensor());
  }

 private:
  TFLM::TfLiteDepthwiseConvParams params;
};

template <typename Tout>
QuantizedDepthwiseSeparableConvOperator<Tout>::QuantizedDepthwiseSeparableConvOperator(TFLM::TfLiteDepthwiseConvParams& _params)
      : params(_params) {}

template <typename Tout>
QuantizedDepthwiseSeparableConvOperator<Tout>& QuantizedDepthwiseSeparableConvOperator<Tout>::set_params(
      TFLM::TfLiteDepthwiseConvParams& _params) {
    params = _params;
    return *this;
}

template <typename Tout>
void QuantizedDepthwiseSeparableConvOperator<Tout>::calculateOpData(TFLM::DWSConvOpData* data) {  // assume kTfLiteInt8

    // int channels_out = SizeOfDimension(filter, 3);
    int channels_out = inputs[filter].tensor()->get_shape()[3];
    // int width = SizeOfDimension(input, 2);
    int width = inputs[in].tensor()->get_shape()[2];
    // int height = SizeOfDimension(input, 1);
    int height = inputs[in].tensor()->get_shape()[1];
    // int filter_width = SizeOfDimension(filter, 2);
    int filter_width = inputs[filter].tensor()->get_shape()[2];
    // int filter_height = SizeOfDimension(filter, 1);
    int filter_height = inputs[filter].tensor()->get_shape()[1];

    int unused_output_height, unused_output_width;
    data->padding = TFLM::ComputePaddingHeightWidth(
        params.stride_height, params.stride_width, 1, 1, height, width,
        filter_height, filter_width, params.padding, &unused_output_height,
        &unused_output_width);

    int num_channels =
        inputs[filter].tensor()->get_shape()[TFLM::kDepthwiseConvQuantizedDimension];
    QuantizationParams affine_quantization =
        inputs[filter].tensor()->get_quantization_params();
    const bool is_per_channel = affine_quantization.num_channels() > 1;

    if (is_per_channel) {
      //  Currently only Int8 is supported for per channel quantization.
      // TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
      // TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);
      if(!(affine_quantization.num_channels() == num_channels)){
        Context::get_default_context()->throwError(new qDwsConvPerChannelMismatchError);
      }
      if(!(num_channels ==
          inputs[filter].tensor()->get_shape()
              [affine_quantization
                   .num_channels()])) {  // FIXME:
                                       // affine_quantization.num_channels()-1?
        Context::get_default_context()->throwError(new qDwsConvPerChannelMismatchError);
      }
    }

    // Populate multiplier and shift using affine quantization.
    // FIXME: where does params.scale comes from? vs. quantization.params
    // const float input_scale = input->params.scale;
    const float input_scale =
        inputs[in].tensor()->get_quantization_params().get_scale_for_channel(0);
    // const float output_scale = output->params.scale;
    const float output_scale =
        outputs[out].tensor()->get_quantization_params().get_scale_for_channel(0);

    for (int i = 0; i < num_channels; ++i) {
      // If per-tensor quantization parameter is specified, broadcast it along
      // the quantization dimension (channels_out).
      const float scale = is_per_channel ? inputs[filter]
                                               .tensor()
                                               ->get_quantization_params()
                                               .get_scale_for_channel(i)
                                         : inputs[filter]
                                               .tensor()
                                               ->get_quantization_params()
                                               .get_scale_for_channel(0);
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      TFLM::QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      reinterpret_cast<int32_t*>(data->per_channel_output_multiplier)[i] =
          significand;
      reinterpret_cast<int32_t*>(data->per_channel_output_shift)[i] =
          channel_shift;

      /*
      // Populate scalar quantization parameters.
      // This check on legacy quantization parameters is kept only for backward
      // compatibility.
      if (input->type == kTfLiteUInt8) {
        // Check bias scale == input scale * filter scale.
        double real_multiplier = 0.0;
        TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
            context, input, filter, bias, output, &real_multiplier));
        int exponent;

        // Populate quantization parameteters with multiplier and shift.
        QuantizeMultiplier(real_multiplier, multiplier, &exponent);
        *shift = -exponent;
      }
      */
      // if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
      TFLM::CalculateActivationRangeQuantized(
          params.activation, outputs[out].tensor(),
          &data->output_activation_min, &data->output_activation_max);
    }
  }

}  // namespace uTensor
#endif
