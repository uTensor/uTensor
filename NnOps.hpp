#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include "quantization_utils.hpp"
#include "tensor.hpp"

template <class TIn, class T2, class TOut>
void Relu(Tensor* input, Tensor* in_min, Tensor* in_max,
          Tensor* output, Tensor* out_min, Tensor* out_max) {
  const float input_min = in_min->read<T2>(0, 0)[0];
  const float input_max = in_max->read<T2>(0, 0)[0];
  TIn* in = input->read<TIn>(0, 0);

  const TOut min_as_quantized =
      FloatToQuantized<TOut>(0.0f, input_min, input_max);
  TOut* out = output->write<TOut>(0, 0);
  for (uint32_t i = 0; i < output->getSize(); i++) {
    if (in[i] > min_as_quantized) {
      out[i] = in[i];
    } else {
      out[i] = min_as_quantized;
    }
  }
  T2* v_out_min = out_min->write<T2>(0, 0);
  *v_out_min = input_min;
  T2* v_out_max = out_max->write<T2>(0, 0);
  *v_out_max = input_max;
}
#endif  // UTENSOR_NN_OPS
