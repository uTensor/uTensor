#ifndef UTENSOR_ARRAY_OPS
#define UTENSOR_ARRAY_OPS

#include <limits>
#include <algorithm>
#include <math.h>
#include "uTensor_util.hpp"

//quantization_utils.h:181
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

//T = inferred
//mode = MIN_FIRST
//name = unspecified
template <typename T>
void QuantizeV2(Tensor<float> input, Tensor<float> _min_range, Tensor<float> _max_range,
                    Tensor<T> output, Tensor<float> output_min, Tensor<float> output_max) {

    float input_min_range = *(_min_range.getPointer({0}));
    float input_max_range = *(_max_range.getPointer({0}));

    if(input_max_range < input_min_range) ERR_EXIT("input_max_range must be larger than input_min_range.");

    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                   fabsf(input_max_range))) / 100.0f;

    float max_range = std::max(input_max_range, min_range + epsilon);
    max_range = std::max(0.0f, max_range);

    FloatToQuantizedStruct<T> f2q(min_range, max_range);

    //quantization_utils.h:149
    float* input_ptr = input.getPointer({});
    T* output_ptr = output.getPointer({});
    float* output_min_ptr = output_min.getPointer({0});
    float* output_max_ptr = output_max.getPointer({0});

    ///NT: need error checking at some point...
    for(uint32_t i = 0; i < input.getSize(); i++) {
        float val = std::round(input_ptr[i] * f2q.range_scale);
        val -= f2q.range_min_scaled - f2q.lowest_quantized();
        val = std::max(val, f2q.lower_bound_float());
        val = std::min(val, f2q.upper_bound_float());
        uint32_t intTmp = static_cast<uint32_t>(val); ///NT: omit this?
        output_ptr[i] = static_cast<T>(intTmp);
    }

    *output_min_ptr = min_range;
    *output_max_ptr = max_range;
    
}

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

//mode = MIN_FIRST
//name = unspecified
//dequantize_op.cc: 87
template <typename T>
void dequantize(Tensor<T> input, Tensor<float> min_range, Tensor<float> max_range, Tensor<float> output) {
    float min = *(min_range.getPointer({0}));
    float max = *(max_range.getPointer({0}));
    T* input_ptr = input.getPointer({});
    float* output_ptr = output.getPointer({});

    //quantization_utils.h: 771
    QuantizedToFloatStruct<T> q2f(min, max);

    //quantization_utils.h: 141
    for(uint32_t i = 0; i < input.getSize(); i++) {
        float val = static_cast<float>(input_ptr[i]);
        output_ptr[i] = ((q2f.range_min_rounded - q2f.lowest_quantized() * q2f.range_scale) + \
                        val * q2f.range_scale);
    }

}

#endif  //UTENSOR_ARRAY_OPS