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
  QuantizedDepthwiseSeparableConvOperator(TFLM::TfLitePadding const &param_padding,
                      int const &stride_width, int const &stride_height, int const &depth_multiplier,
                      TFLM::TfLiteFusedActivation const &activation, int const &dilation_width_factor,
                      int const &dilation_height_factor);


  void calculateOpData(TFLM::TfLitePaddingValues &padding, int32_t &output_multiplier,
                        int output_shift, int32_t *per_channel_output_multiplier,
                        int32_t *per_channel_output_shift, int32_t &output_activation_min,
                        int32_t &output_activation_max, TFLM::TfLitePadding const param_padding,
                      int const stride_width, int const stride_height, int const depth_multiplier,
                      TFLM::TfLiteFusedActivation const activation, int const dilation_width_factor,
                      int const dilation_height_factor);


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

  
    TFLM::DepthwiseParams op_params;
    TFLM::TfLitePaddingValues padding;
    calculateOpData(padding, output_multiplier,
                    output_shift, per_channel_output_multiplier,
                    per_channel_output_shift, output_activation_min,
                    output_activation_max, param_padding,
                    stride_width, stride_height, depth_multiplier,
                    activation, dilation_width_factor,
                    dilation_height_factor);
    op_params.padding_type = TFLM::PaddingType::kSame;
    op_params.padding_values.width = padding.width;
    op_params.padding_values.height = padding.height;
    op_params.stride_width = stride_width;
    op_params.stride_height = stride_height;
    op_params.dilation_width_factor = dilation_width_factor;
    op_params.dilation_height_factor = dilation_height_factor;
    op_params.depth_multiplier = depth_multiplier;
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

    TFLM::DepthwiseConvPerChannel<Tout>(op_params, per_channel_output_multiplier,
                                  per_channel_output_shift,
                                  inputs[in].tensor(), inputs[filter].tensor(),
                                  inputs[bias].tensor(), outputs[out].tensor());
  }

 private:
 //TfLiteDepthwiseConvParams
 //Set by constructors
  TFLM::TfLitePadding param_padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TFLM::TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
  //DWSConvOpData (check the previous commit)
  TFLM::TfLitePaddingValues padding;
  int32_t output_multiplier;
  int output_shift;
  int32_t per_channel_output_multiplier[TFLM::kMaxChannels]; // TODO PUT this stupid shit in the allocator
  int32_t per_channel_output_shift[TFLM::kMaxChannels];
  int32_t output_activation_min;
  int32_t output_activation_max;
};

template <typename Tout>
QuantizedDepthwiseSeparableConvOperator<Tout>::QuantizedDepthwiseSeparableConvOperator(TFLM::TfLitePadding const &_param_padding,
                      int const &_stride_width, int const &_stride_height, int const &_depth_multiplier,
                      TFLM::TfLiteFusedActivation const &_activation, int const &_dilation_width_factor,
                      int const &_dilation_height_factor)
                      : param_padding(_param_padding), stride_width(_stride_width),
                      stride_height(_stride_height), depth_multiplier(_depth_multiplier),
                      activation(_activation), dilation_width_factor(_dilation_width_factor),
                      dilation_height_factor(_dilation_height_factor) {}

template <typename Tout>
void QuantizedDepthwiseSeparableConvOperator<Tout>::calculateOpData(TFLM::TfLitePaddingValues &padding, int32_t &output_multiplier,
                      int output_shift, int32_t *per_channel_output_multiplier,
                      int32_t *per_channel_output_shift, int32_t &output_activation_min,
                      int32_t &output_activation_max, TFLM::TfLitePadding const param_padding,
                      int const stride_width, int const stride_height, int const depth_multiplier,
                      TFLM::TfLiteFusedActivation const activation, int const dilation_width_factor,
                      int const dilation_height_factor) {
    int channels_out = inputs[filter].tensor()->get_shape()[3];
    int width = inputs[in].tensor()->get_shape()[2];
    int height = inputs[in].tensor()->get_shape()[1];
    int filter_width = inputs[filter].tensor()->get_shape()[2];
    int filter_height = inputs[filter].tensor()->get_shape()[1];

    int unused_output_height, unused_output_width;

    padding = TFLM::ComputePaddingHeightWidth(
        stride_height, stride_width, 1, 1, height, width,
        filter_height, filter_width, param_padding, &unused_output_height,
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
      reinterpret_cast<int32_t*>(per_channel_output_multiplier)[i] =
          significand;
      reinterpret_cast<int32_t*>(per_channel_output_shift)[i] =
          channel_shift;

      // if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
      TFLM::CalculateActivationRangeQuantized(
          activation, outputs[out].tensor(),
          &output_activation_min, &output_activation_max);
    }

  }

}  // namespace uTensor
#endif
