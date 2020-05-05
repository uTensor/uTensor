#include "symmetric_quantization_utils.hpp"

namespace uTensor {

DEFINE_ERROR(SymmetricQuantizationFixedPointError);
DEFINE_ERROR(SymmetricQuantizationFixedPointRangeError);
DEFINE_ERROR(InvalidFCQuantizationScalesError);
DEFINE_ERROR(FCQuantizationScaleMultipleLTzeroError);

namespace TFLM {

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
  static_assert(8 == sizeof(double), "Invalid size of double in IntegerFrExp");

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
}

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
  if(!(q_fixed <= (1ll << 31))){
    Context::get_default_context()->throwError(new SymmetricQuantizationFixedPointError);
  }
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  if(!(q_fixed < std::numeric_limits<int32_t>::max())){
    Context::get_default_context()->throwError(new SymmetricQuantizationFixedPointRangeError);
  }
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

void CalculateActivationRangeQuantizedImpl(TfLiteFusedActivation activation,
                                           int32_t qmin, int32_t qmax,
                                           Tensor& output, int32_t* act_min,
                                           int32_t* act_max) {
  // const auto scale = output->params.scale;
  const auto scale = output->get_quantization_params().get_scale_for_channel(0);
  // const auto zero_point = output->params.zero_point;
  const auto zero_point =
      output->get_quantization_params().get_zeroP_for_channel(0);

  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
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

void CalculateActivationRangeQuantized(TfLiteFusedActivation activation,
                                       Tensor& output, int32_t* act_min,
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

// Following two functions are borrowed from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/fully_connected.cc for operator compatibility
void GetQuantizedConvolutionMultipler(const Tensor& input,
                                      const Tensor& filter,
                                      const Tensor& bias,
                                      Tensor& output,
                                      double* multiplier) {
  const double input_product_scale = static_cast<double>(input->get_quantization_params().get_scale_for_channel(0)) *
                                     static_cast<double>(filter->get_quantization_params().get_scale_for_channel(0));
  // TODO(ahentz): The following conditions must be guaranteed by the training
  // pipeline.
  if (bias) {
    const double bias_scale = static_cast<double>(bias->get_quantization_params().get_scale_for_channel(0));
      if(!(std::abs(input_product_scale - bias_scale) <=
                       1e-6 * std::min(input_product_scale, bias_scale))){
        Context::get_default_context()->throwError( new InvalidFCQuantizationScalesError );
      }
  }
  return GetQuantizedConvolutionMultipler(input, filter, output,
                                          multiplier);
}

void GetQuantizedConvolutionMultipler(const Tensor& input,
                                      const Tensor& filter,
                                      Tensor& output,
                                      double* multiplier) {
  const double input_product_scale =
      static_cast<double>(input->get_quantization_params().get_scale_for_channel(0) * filter->get_quantization_params().get_scale_for_channel(0));
  if(!(input_product_scale >= 0)){
    Context::get_default_context()->throwError( new FCQuantizationScaleMultipleLTzeroError );
  }
  *multiplier = input_product_scale / static_cast<double>(output->get_quantization_params().get_scale_for_channel(0));

  return;
}

int32_t MultiplyByQuantizedMultiplier(int32_t acc, int32_t output_multiplier, int32_t output_shift) {
  // simplified MultiplyByQuantizedMultiplier, may introduce rounding
  // error
  int left_shift = output_shift > 0
                        ? output_shift
                        : 0;
  int right_shift = output_shift > 0
                        ? 0
                        : -output_shift;
  acc = ((acc * (1 << left_shift)) *
          output_multiplier) >>
        right_shift;
  return acc;
}

}
}
