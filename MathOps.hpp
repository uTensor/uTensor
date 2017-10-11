#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include <climits>
#include "quantization_utils.hpp"
#include "tensor.hpp"

void CalculateUsedRange(Tensor<int>& input, int32_t* used_min_quan,
                        int32_t* used_max_quan) {
  int32_t minimum = INT_MAX;
  int32_t maxmum = INT_MIN;
  uint32_t size = input.getSize();
  int* in_ptr = input.getPointer({});

  for (uint32_t i = 0; i < size; i++) {
    if (minimum > in_ptr[i]) minimum = static_cast<int32_t>(in_ptr[i]);

    if (maxmum < in_ptr[i]) maxmum = static_cast<int32_t>(in_ptr[i]);
  }

  *used_min_quan = minimum;
  *used_max_quan = maxmum;
}
template<class T1, class T2>
void Requantization_Range(Tensor<T1> input, Tensor<T2> min, Tensor<T2> max, Tensor<T2> out_min,
                          Tensor<T2> out_max) {
  const float input_min = *(min.getPointer({}));
  const float input_max = *(max.getPointer({}));

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange(input, &used_min_quan, &used_max_quan);

  const float used_min = std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);


  float* c_min = out_min.getPointer({});
  *c_min = used_min;
  float* c_max = out_max.getPointer({});
  *c_max = used_max;
}

#endif  // UTENSOR_MATH_OPS
