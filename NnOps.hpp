#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include "quantization_utils.hpp"
#include "tensor.hpp"

template <class TIn, class T2, class TOut>
void Relu(Tensor<TIn> input, Tensor<T2> in_min, Tensor<T2> in_max,
          Tensor<TOut> output, Tensor<T2> out_min, Tensor<T2> out_max) {
  const float input_min = in_min.getPointer({})[0];
  const float input_max = in_max.getPointer({})[0];
  TIn* in = input.getPointer({});

  const TOut min_as_quantized =
      FloatToQuantized<TOut>(0.0f, input_min, input_max);
  TOut* out = output.getPointer({});
  for (uint32_t i = 0; i < output.getSize(); i++) {
    if (in[i] > min_as_quantized) {
      out[i] = in[i];
    } else {
      out[i] = min_as_quantized;
    }
  }
  T2* v_out_min = out_min.getPointer({});
  *v_out_min = input_min;
  T2* v_out_max = out_max.getPointer({});
  *v_out_max = input_max;
}
#endif  // UTENSOR_NN_OPS
