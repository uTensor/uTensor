#ifndef UTENSOR_QUANT_UTILS
#define UTENSOR_QUANT_UTILS

#include <math.h>
#include <limits>
#include "tensor.hpp"

// reference: quantization_utils.h:181
template <class T>
int64_t FloatToQuantizedUnclamped(float input, float range_min,
                                  float range_max) {
  const int64_t lowest_quantized =
      static_cast<double>(std::numeric_limits<T>::lowest());
  if (range_min == range_max) {
    return lowest_quantized;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64_t quantized =
      (round(input * range_scale) - round(range_min * range_scale));
  quantized += lowest_quantized;

  return quantized;
}

template <class T>
float QuantizedToFloat(T input, float range_min, float range_max) {
  if (std::is_same<T, float>::value) {
    return input;
  }
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64_t lowest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::min());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double range_min_rounded =
      round(range_min / static_cast<float>(range_scale)) *
      static_cast<float>(range_scale);
  const double result = range_min_rounded + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T>
T FloatToQuantized(float input, float range_min, float range_max) {
  if (std::is_same<T, float>::value) {
    return input;
  }
  int64_t quantized = FloatToQuantizedUnclamped<T>(input, range_min, range_max);
  const int64_t lowest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::min());
  const int64_t highest_quantized =
      static_cast<int64_t>(std::numeric_limits<T>::max());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32_t>(quantized));
}

template <class T1, class T2>
inline void RequantizeManyInNewRange(Tensor<T1> input, uint32_t count,
                                     float min_input, float max_input,
                                     float min_output, float max_output,
                                     Tensor<T2> output) {
  T1 *in_ptr = input.getPointer({});
  T2 *out_ptr = output.getPointer({});
  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(in_ptr[index], min_input, max_input);
    out_ptr[index] = FloatToQuantized<T2>(input_float, min_output, max_output);
  }
}

//quantization_utils.h : 239
void RequantizeManyInNewRangeReference(const int* input, int32_t count,
    float min_input, float max_input,
    float min_output,
    float max_output,
    unsigned char* output); 

template <typename T>
struct FloatToQuantizedStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64_t number_of_steps = static_cast<int64_t>(1)
                                             << number_of_bits;
  static constexpr double range_adjust =
      (number_of_steps / (number_of_steps - 1.0));

  // Casting QInt32's lowest or highest to a float gives a float that can't be
  // cast back to int32 or QInt32.  Instead, use bounds that can be converted
  // back to int32 without going outside the range of an int32.
  static float lower_bound_float() {
    return std::max(static_cast<float>(std::numeric_limits<T>::lowest()),
                    -2.147483648e+09f);
  }
  static float upper_bound_float() {
    return std::min(static_cast<float>(std::numeric_limits<T>::max()),
                    +2.147483520e+09f);
  }

  static float lowest_quantized() {
    return static_cast<float>(std::numeric_limits<T>::lowest());
  }

  FloatToQuantizedStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale(range_max == range_min
                        ? 0.0
                        : (number_of_steps - 1.0) / (range_max - range_min)),
        range_min_scaled(round(range_min * range_scale)) {}

  const float range_min;
  const float range_scale;
  const float range_min_scaled;
};

template <typename T>
struct QuantizedToFloatStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64_t number_of_steps = static_cast<int64_t>(1)
                                             << number_of_bits;

  static float lowest_quantized() {
    return static_cast<float>(std::numeric_limits<T>::lowest());
  }

  QuantizedToFloatStruct(float range_min, float range_max)
      : range_min(range_min),
        range_scale((range_max - range_min) / (number_of_steps - 1.0)),
        range_min_rounded(range_max == range_min
                              ? range_min
                              : round(range_min / range_scale) * range_scale) {}

  const float range_min;
  const float range_scale;
  const float range_min_rounded;
};

#endif  // UTENSOR_QUANT_UTILS
