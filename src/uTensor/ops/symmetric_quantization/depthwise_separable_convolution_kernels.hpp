#ifndef UTENSOR_S_QUANTIZED_DWS_OPS_KERNELS_H
#define UTENSOR_S_QUANTIZED_DWS_OPS_KERNELS_H
#include "uTensor/core/context.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/types.hpp"
#include "uTensor/core/uTensor_util.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {
namespace TFLM {

DECLARE_ERROR(qDwsMismatchedDimensionsError);
DECLARE_ERROR(InvalidQDwsActivationRangeError);
DECLARE_ERROR(InvalidQDwsOutputDepthError);

typedef enum {
  kTfLitePaddingUnknown = 0,
  kTfLitePaddingSame,
  kTfLitePaddingValid,
} TfLitePadding;

struct TfLiteDepthwiseConvParams {
  // Parameters for DepthwiseConv version 1 or above.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  // `depth_multiplier` is redundant. It's used by CPU kernels in
  // TensorFlow 2.0 or below, but ignored in versions above.
  //
  // The information can be deduced from the shape of input and the shape of
  // weights. Since the TFLiteConverter toolchain doesn't support partially
  // specified shapes, relying on `depth_multiplier` stops us from supporting
  // graphs with dynamic shape tensors.
  //
  // Note: Some of the delegates (e.g. NNAPI, GPU) are still relying on this
  // field.
  int depth_multiplier;
  TfLiteFusedActivation activation;
  // Parameters for DepthwiseConv version 2 or above.
  int dilation_width_factor;
  int dilation_height_factor;
};

enum PaddingType : uint8_t { kNone = 0, kSame, kValid };

struct PaddingValues {
  int16_t width;
  int16_t height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
};
struct DepthwiseParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  const int32_t* output_multiplier_per_channel;
  const int32_t* output_shift_per_channel;
};

struct TfLitePaddingValues {
  int32_t width;
  int32_t height;
  int width_offset;
  int height_offset;
};

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
inline int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int total_padding =
      ((out_size - 1) * stride + effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = total_padding % 2;
  return total_padding / 2;
}

// Matching GetWindowedOutputSize in TensorFlow.
inline int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate = 1) {
  int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding) {
    case kTfLitePaddingSame:
      return (image_size + stride - 1) / stride;
    case kTfLitePaddingValid:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      return 0;
  }
}

TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width);

void ComputePaddingHeightWidth(int stride_height, int stride_width,
                               int dilation_rate_height,
                               int dilation_rate_width, int in_height,
                               int in_width, int filter_height,
                               int filter_width, int32_t* padding_height,
                               int32_t* padding_width, TfLitePadding padding,
                               int* out_height, int* out_width);

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1);

// Only support int8 for now
void DepthwiseConvPerChannel(const DepthwiseParams& params,
                             const int32_t* output_multiplier,
                             const int32_t* output_shift, Tensor& input,
                             Tensor& filter, Tensor& bias, Tensor& output);

}  // namespace TFLM
}  // namespace uTensor
#endif
