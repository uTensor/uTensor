#ifndef UTENSOR_CONVOLUTION_OPS_H
#define UTENSOR_CONVOLUTION_OPS_H
#include <algorithm>
#include <cmath>
#include <limits>

#include "Convolution_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {

// Can use these intermediate types to make the convolution operator more
// generic. Maxpool, conv, average pool, median etc. are all basically the same
// operation with target functions.
template <typename T>
class ConvFilter {
  T tmp;
  const Tensor& filter;

 public:
  ConvFilter(const Tensor& filter) : filter(filter) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    const T filter_value = filter(i, j, k, l);
    tmp += (input_value * filter_value);
  }
  inline T finalize() const { return tmp; }
  inline const int16_t height() const { return filter->get_shape()[0]; }
  inline const int16_t width() const { return filter->get_shape()[1]; }
  inline const int16_t in_channels() const { return filter->get_shape()[2]; }
  inline const int16_t out_channels() const { return filter->get_shape()[3]; }
};

template <typename T>
class MaxFilter {
  T tmp;
  int16_t h;
  int16_t w;
  int16_t c;

 public:
  MaxFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = std::numeric_limits<T>::lowest(); }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp = std::max(tmp, input_value);
  }
  inline T finalize() const { return tmp; }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

template <typename T>
class MinFilter {
  T tmp;
  int16_t h;
  int16_t w;
  int16_t c;

 public:
  MinFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = std::numeric_limits<T>::max(); }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp = std::min(tmp, input_value);
  }
  inline T finalize() const { return tmp; }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

template <typename T>
class AvgFilter {
  T tmp;
  int16_t w;
  int16_t h;
  int16_t c;

 public:
  AvgFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp += input_value;
  }
  inline T finalize() const {
    return tmp / (w * h);
  }  //(static_cast<T>(w*h)); }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

template <typename T>
class ConvOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { in, filter };
  enum names_out : uint8_t { out };
  ConvOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    ConvFilter<T> conv(inputs[filter].tensor());
    generic_convolution_kernel<T, ConvFilter<T>>(
        outputs[out].tensor(), inputs[in].tensor(), conv, _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, depthwise_filter, pointwise_filter };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  DepthwiseSeparableConvOperator(std::initializer_list<uint16_t> strides,
                                 Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    TensorShape& df_shape = inputs[depthwise_filter].tensor()->get_shape();
    TensorShape& pf_shape = inputs[pointwise_filter].tensor()->get_shape();
    TensorShape& out_shape = outputs[out].tensor()->get_shape();

    if (in_shape[3] != df_shape[2]) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    if (pf_shape[0] != 1 || pf_shape[1] != 1) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    depthwise_separable_convolution_kernel<T>(
        outputs[out].tensor(), inputs[in].tensor(),
        inputs[depthwise_filter].tensor(), inputs[pointwise_filter].tensor(),
        _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T, typename Filter>
class GenericPoolOp : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  GenericPoolOp(std::initializer_list<uint16_t> k_size,
                std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
    i = 0;
    for (auto k : k_size) {
      _k_size[i++] = k;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    Filter filter(_k_size[0], _k_size[1], in_shape[3]);
    generic_pool_convolution_kernel<T, Filter>(
        outputs[out].tensor(), inputs[in].tensor(), filter, _padding, _stride);
  }

 private:
  uint16_t _k_size[2];
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
using MaxPoolOp = GenericPoolOp<T, MaxFilter<T>>;

template <typename T>
using AvgPoolOp = GenericPoolOp<T, AvgFilter<T>>;

namespace TFLM {

typedef enum {
  kTfLitePaddingUnknown = 0,
  kTfLitePaddingSame,
  kTfLitePaddingValid,
} TfLitePadding;

typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActRelu1,2  // min(max(-1, x), 1)
  kTfLiteActRelu6,  // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

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
} ;

enum PaddingType : uint8_t { kNone, kSame, kValid };

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
  int width;
  int height;
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

inline TfLitePaddingValues ComputePaddingHeightWidth(
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

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1) {
  assert(s0[i0] == s1[i1]);
  return s0[i0];
}

#define DCHECK_LE(a,b) assert(a <= b)
#define DCHECK_EQ(a,b) assert(a == b)
#define DCHECK(a) assert(a)

inline void DepthwiseConvPerChannel(
const DepthwiseParams& params, const int32_t* output_multiplier, const int32_t* output_shift,
Tensor& input, Tensor& filter, Tensor& bias, Tensor& output
) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TensorShape input_shape = input->get_shape();
  TensorShape filter_shape = filter->get_shape();
  TensorShape output_shape = output->get_shape();
  

  assert(input_shape.num_dims() == 4);
  assert(filter_shape.num_dims() == 4);
  assert(output_shape.num_dims() == 4);

  DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const int filter_height = filter_shape[1];
  const int filter_width = filter_shape[2];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  DCHECK_EQ(bias->get_shape().get_linear_size(), (uint16_t) output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  // int32_t input_val = input_data[Offset(input_shape, batch, in_y,
                  //                                     in_x, in_channel)];
                  int32_t input_val = input(batch, in_y, in_x, in_channel);
                  // int32_t filter_val = filter_data[Offset(
                  //     filter_shape, 0, filter_y, filter_x, output_channel)];
                  int32_t filter_val = filter(batch, filter_y, filter_x, output_channel);
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8, even though it
                  // is represented using int32.
                  // int32 += int8 * (int8 - int8) so the highest value we can
                  // get from each accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
            //assuming bias data will always be provided
            acc += (int32_t) bias(output_channel);
            // acc = MultiplyByQuantizedMultiplier(
            //     acc, output_multiplier[output_channel],
            //     output_shift[output_channel]);

            //simplified MultiplyByQuantizedMultiplier, may introduce rounding error
            int left_shift = output_shift[output_channel] > 0 ? output_shift[output_channel] : 0;
            int right_shift = output_shift[output_channel] > 0 ? 0 : -output_shift[output_channel];
            acc = ((acc * (1 << left_shift)) * output_multiplier[output_channel]) >> right_shift;

            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output(batch, out_y, out_x,
                               output_channel) = static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}

typedef struct TfLiteAffineQuantization {
  TfLiteFloatArray* scale;
  TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;


constexpr uint64_t kSignMask = 0x8000000000000000LL;
constexpr uint64_t kExponentMask = 0x7ff0000000000000LL;
constexpr int32_t kExponentShift = 52;
constexpr int32_t kExponentBias = 1023;
constexpr uint32_t kExponentIsBadNum = 0x7ff;
constexpr uint64_t kFractionMask = 0x000fffffffc00000LL;
constexpr uint32_t kFractionShift = 22;
constexpr uint32_t kFractionRoundingMask = 0x003fffff;
constexpr uint32_t kFractionRoundingThreshold = 0x00200000;

int64_t IntegerFrExp(double input, int* shift) {
  // Make sure our assumptions about the double layout hold.
  DCHECK_EQ(8, sizeof(double));

  // We want to access the bits of the input double value directly, which is
  // tricky to do safely, so use a union to handle the casting.
  union {
    double double_value;
    uint64_t double_as_uint;
  } cast_union;
  cast_union.double_value = input;
  const uint64_t u = cast_union.double_as_uint;

  // If the bitfield is all zeros apart from the sign bit, this is a normalized
  // zero value, so return standard values for this special case.
  if ((u & ~kSignMask) == 0) {
    *shift = 0;
    return 0;
  }

  // Deal with NaNs and Infs, which are always indicated with a fixed pattern in
  // the exponent, and distinguished by whether the fractions are zero or
  // non-zero.
  const uint32_t exponent_part = ((u & kExponentMask) >> kExponentShift);
  if (exponent_part == kExponentIsBadNum) {
    *shift = std::numeric_limits<int>::max();
    if (u & kFractionMask) {
      // NaN, so just return zero (with the exponent set to INT_MAX).
      return 0;
    } else {
      // Infinity, so return +/- INT_MAX.
      if (u & kSignMask) {
        return std::numeric_limits<int64_t>::min();
      } else {
        return std::numeric_limits<int64_t>::max();
      }
    }
  }

constexpr int kDepthwiseConvQuantizedDimension = 3;
void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier,
                        int* shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
#ifdef TFLITE_EMULATE_FLOAT
  // If we're trying to avoid the use of floating-point instructions (for
  // example on microcontrollers) then use an alternative implementation
  // that only requires integer and bitwise operations. To enable this, you
  // need to set the define during the build process for your platform.
  int64_t q_fixed = IntegerFrExp(double_multiplier, shift);
#else   // TFLITE_EMULATE_FLOAT
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1ll << 31)));
#endif  // TFLITE_EMULATE_FLOAT
  DCHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  DCHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActRelu1,  // min(max(-1, x), 1)
  kTfLiteActRelu6,  // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

void CalculateActivationRangeQuantizedImpl(TfLiteFusedActivation activation,
                                           int32_t qmin, int32_t qmax,
                                           Tensor& output,
                                           int32_t* act_min, int32_t* act_max) {
  //const auto scale = output->params.scale;
  const auto scale = output->get_quantization_params().get_scale_for_channel(0);
  //const auto zero_point = output->params.zero_point;
  const auto zero_point = output->get_quantization_params().get_zeroP_for_channel(0);

  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(TfLiteRound(f / scale));
  };

  if (activation == kTfLiteActRelu) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  } else if (activation == kTfLiteActRelu6) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  } else if (activation == kTfLiteActRelu1) {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
}

void CalculateActivationRangeQuantized(
                                      TfLiteFusedActivation activation,
                                      Tensor& output,
                                      int32_t* act_min,
                                      int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  // if (output->type == kTfLiteUInt8) {
  //   qmin = std::numeric_limits<uint8_t>::min();
  //   qmax = std::numeric_limits<uint8_t>::max();
  // } else if (output->type == kTfLiteInt8) {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
  // } else if (output->type == kTfLiteInt16) {
  //   qmin = std::numeric_limits<int16_t>::min();
  //   qmax = std::numeric_limits<int16_t>::max();
  // } else {
  //   assert();
  // }

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
}

template <typename T>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

  DepthwiseSeparableConvOperator& set_params(TfLiteDepthwiseConvParams&& _params) {
    param = _params;
  }

  void calculateOpData() {
    param.padding = ComputePaddingHeightWidth(
      params.stride_height, params.stride_width, 1, 1, height, width,
      filter_height, filter_width, params.padding, &unused_output_height,
      &unused_output_width);

    int num_channels = inputs[filter].tensor()->get_shape()[kDepthwiseConvQuantizedDimension];
    QuantizationParams affine_quantization = inputs[filter].tensor()->get_quantization_params();
    const bool is_per_channel = affine_quantization->num_channels() > 1;

    if (is_per_channel) {
    //  Currently only Int8 is supported for per channel quantization.
    // TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    // TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(affine_quantization->num_channels(), num_channels);
    TF_LITE_ENSURE_EQ(num_channels,
        filter->get_shape()[affine_quantization.num_channels()]);  //FIXME: affine_quantization.num_channels()-1?
    }

    // Populate multiplier and shift using affine quantization.
    //FIXME: where does params.scale comes from? vs. quantization.params
    //const float input_scale = input->params.scale;
    const float input_scale = inputs[in].tensor()->get_quantization_params().get_scale_for_channel(0);
    //const float output_scale = output->params.scale;
    const float output_scale = outputs[out].tensor()->get_quantization_params().get_scale_for_channel(0);

    for (int i = 0; i < num_channels; ++i) {
      // If per-tensor quantization parameter is specified, broadcast it along the
      // quantization dimension (channels_out).
      const float scale = is_per_channel ?  inputs[filter].tensor()->get_quantization_params().get_scale_for_channel(i) :  inputs[filter].tensor()->get_quantization_params().get_scale_for_channel(0);
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      per_channel_multiplier[i] = significand;
      per_channel_shift[i] = channel_shift;

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
      //if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
      CalculateActivationRangeQuantized(
            context, activation, outputs[out].tensor(), output_activation_min,
            output_activation_max);
    }

  }

  //FIXME: remove this method
  DepthwiseSeparableConvOperator& set_params(DepthwiseParams&& _params, const int32_t&& _output_multiplier, const int32_t&& _output_shift) {
    params = _params;
    output_multiplier = _output_multiplier;
    output_shift = _output_shift;
    return *this;
  }

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

    DepthwiseConvPerChannel( 
      params, &output_multiplier, &output_shift,
      inputs[in].tensor(), inputs[filter].tensor(), inputs[bias].tensor(), outputs[out].tensor()
    );
  }

private:
  DepthwiseParams params;
  int32_t output_multiplier;
  int32_t output_shift;
};

}  // namespace TFLM
}  // namespace uTensor
#endif
