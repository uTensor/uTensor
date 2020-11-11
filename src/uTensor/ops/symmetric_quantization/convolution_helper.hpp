#ifndef UTENSOR_S_QUANTIZED_CONV_OPS_HELPER_H
#define UTENSOR_S_QUANTIZED_CONV_OPS_HELPER_H
#include "uTensor/core/context.hpp"
#include "symmetric_quantization_utils.hpp"

namespace uTensor {

DECLARE_ERROR(qConvMismatchedDimensionsError);

namespace TFLM {

typedef enum {
  kTfLitePaddingUnknown = 0,
  kTfLitePaddingSame,
  kTfLitePaddingValid,
} TfLitePadding;


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

struct TfLitePaddingValues {
  int32_t width;
  int32_t height;
  int width_offset;
  int height_offset;
};

// It's not guaranteed that padding is symmetric. It's important to keep
// offset for algorithms need all paddings.
int ComputePaddingWithOffset(int stride, int dilation_rate, int in_size,
                                    int filter_size, int out_size,
                                    int* offset);

// Matching GetWindowedOutputSize in TensorFlow.
int ComputeOutSize(TfLitePadding padding, int image_size,
                          int filter_size, int stride, int dilation_rate = 1);

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

} // TFLM
} // uTensor
#endif
