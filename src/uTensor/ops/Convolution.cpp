#include "uTensor/ops/Convolution.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <>
void Conv2dOperator<int8_t>::compute() {
  bool have_bias = inputs.has(bias);
  ConvFilter<int8_t> conv(inputs[filter].tensor());
  if(have_bias) {
    wBias<int8_t> w_bias(inputs[bias].tensor());
    generic_sq_convolution_kernel<ConvFilter<int8_t>>(
      outputs[out].tensor(), inputs[in].tensor(), conv, w_bias, _padding, _stride);
  } else {
    NoBias<int8_t> no_bias;
    generic_sq_convolution_kernel<ConvFilter<int8_t>>(
      outputs[out].tensor(), inputs[in].tensor(), conv, no_bias, _padding, _stride);

  }
}

}// ReferenceOperators
} // uTensor
