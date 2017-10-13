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
template <class T1, class T2>
void Requantization_Range(Tensor<T1> input, Tensor<T2> min, Tensor<T2> max,
                          Tensor<T2> out_min, Tensor<T2> out_max) {
  const float input_min = *(min.getPointer({}));
  const float input_max = *(max.getPointer({}));

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange(input, &used_min_quan, &used_max_quan);

  const float used_min =
      std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);

  float* c_min = out_min.getPointer({});
  *c_min = used_min;
  float* c_max = out_max.getPointer({});
  *c_max = used_max;
}
template <class T1, class T2, class Toutput>
void Requantize(Tensor<T1> input, Tensor<T2> in_min, Tensor<T2> in_max,
                Tensor<T2> r_min, Tensor<T2> r_max, Tensor<Toutput> output,
                Tensor<T2> out_min, Tensor<T2> out_max) {
  const float input_min = in_min.getPointer({})[0];
  const float input_max = in_max.getPointer({})[0];
  const float r_output_min = r_min.getPointer({})[0];
  const float r_output_max = r_max.getPointer({})[0];

  RequantizeManyInNewRange<T1, Toutput>(input, input.getSize(), input_min,
                                        input_max, r_output_min, r_output_max,
                                        output);
  float* v_out_min = out_min.getPointer({});
  *v_out_min = r_output_min;
  float* v_out_max = out_max.getPointer({});
  *v_out_max = r_output_max;
}
template<class TIn, class TOut>
void Add(Tensor<TIn> input, Tensor<TIn> input2, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const TIn *p_in2 = input2.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = out.getSize();
  for (uint32_t i = 0; i < size; i++) {
    p_out[i] = p_in[i] + p_in2[i];
  }
}
template<class TIn, class Td, class TOut>
void Min(Tensor<TIn> input, Tensor<Td> dim, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const Td *p_in2 = dim.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = dim.getSize();
  vector<uint32_t> shape = input.getShape();
  TOut min = INT_MAX;
  for (uint32_t i = 0; i < size; i++) {
    Td n_dim = p_in2[i];
    uint32_t num = shape[n_dim];
    uint32_t stride = input.getStride(n_dim);
    for (uint32_t j = stride; j < stride + num; j++) {
        if (p_in[j] < min) {
            min = p_in[j];
        }
    }
    p_out[i] = min;
    min = INT_MAX;   
  }
}
template<class TIn, class Td, class TOut>
void Max(Tensor<TIn> input, Tensor<Td> dim, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const Td *p_in2 = dim.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = dim.getSize();
  vector<uint32_t> shape = input.getShape();
  TOut max = INT_MIN;
  for (uint32_t i = 0; i < size; i++) {
    Td n_dim = p_in2[i];
    uint32_t num = shape[n_dim];
    uint32_t stride = input.getStride(n_dim);
    for (uint32_t j = stride; j < stride + num; j++) {
        if (p_in[j] > max) {
            max = p_in[j];
        }
    }
    p_out[i] = max;
    max = INT_MIN;
  }
}
#endif  // UTENSOR_MATH_OPS
