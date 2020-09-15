#include "convolution_helper.hpp"
namespace uTensor {

DEFINE_ERROR(qConvMismatchedDimensionsError);
namespace TFLM {

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
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
int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate) {
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

void ComputePaddingHeightWidth(int stride_height, int stride_width,
                               int dilation_rate_height,
                               int dilation_rate_width, int in_height,
                               int in_width, int filter_height,
                               int filter_width, int32_t* padding_height,
                               int32_t* padding_width, TfLitePadding padding,
                               int* out_height, int* out_width) {
  *out_width = ComputeOutSize(padding, in_width, filter_width, stride_width,
                              dilation_rate_width);
  *out_height = ComputeOutSize(padding, in_height, filter_height, stride_height,
                               dilation_rate_height);

  TfLitePaddingValues padding_values;
  int offset = 0;
  *padding_height =
      ComputePaddingWithOffset(stride_height, dilation_rate_height, in_height,
                               filter_height, *out_height, &offset);
  // padding_values.height_offset = offset;
  *padding_width =
      ComputePaddingWithOffset(stride_width, dilation_rate_width, in_width,
                               filter_width, *out_width, &offset);
  // padding_values.width_offset = offset;
  // return padding_values;
}

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1) {
  if (!(s0[i0] == s1[i1])) {
    Context::get_default_context()->throwError(
        new qConvMismatchedDimensionsError);
  }

  return s0[i0];
}

} // TFLM
} // uTensot
