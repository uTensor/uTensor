#ifndef UTENSOR_QUANT_UTILS
#define UTENSOR_QUANT_UTILS

#include <limits>
#include <math.h>

//reference: quantization_utils.h:181
template <typename T>
struct FloatToQuantizedStruct {
  static constexpr int number_of_bits = sizeof(T) * 8;
  static constexpr int64 number_of_steps = static_cast<int64>(1)
                                           << number_of_bits;
  static constexpr double range_adjust =
      (number_of_steps / (number_of_steps - 1.0));

  // Casting QInt32's lowest or highest to a float gives a float that can't be
  // cast back to int32 or QInt32.  Instead, use bounds that can be converted
  // back to int32 without going outside the range of an int32.
  static float lower_bound_float() {
    return std::max(
        static_cast<float>(std::numeric_limits<T>::lowest()), -2.147483648e+09f);
  }
  static float upper_bound_float() {
    return std::min(
        static_cast<float>(std::numeric_limits<T>::max()), +2.147483520e+09f);
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
  static constexpr int64 number_of_steps = static_cast<int64>(1)
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

#endif  //UTENSOR_QUANT_UTILS