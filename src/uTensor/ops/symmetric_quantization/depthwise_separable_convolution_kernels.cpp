#include "depthwise_separable_convolution_kernels.hpp"

namespace uTensor {
namespace TFLM {

DEFINE_ERROR(qDwsMismatchedDimensionsError);
DEFINE_ERROR(InvalidQDwsActivationRangeError);
DEFINE_ERROR(InvalidQDwsOutputDepthError);

TfLitePaddingValues ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, TfLitePadding padding, int* out_height, int* out_width) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  TfLitePaddingValues padding_values;
  int offset = 0;
  padding_values.height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  padding_values.height_offset = offset;
  padding_values.width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  padding_values.width_offset = offset;
  return padding_values;
}

void ComputePaddingHeightWidth(
    int stride_height, int stride_width, int dilation_rate_height,
    int dilation_rate_width, int in_height, int in_width, int filter_height,
    int filter_width, int* padding_height, int* padding_width, TfLitePadding padding, int* out_height, int* out_width) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  TfLitePaddingValues padding_values;
  int offset = 0;
  *padding_height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  //padding_values.height_offset = offset;
  *padding_width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  //padding_values.width_offset = offset;
  //return padding_values;
}

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1) {
  if (!(s0[i0] == s1[i1])) {
    Context::get_default_context()->throwError(
        new qDwsMismatchedDimensionsError);
  }

  return s0[i0];
}

}
}
