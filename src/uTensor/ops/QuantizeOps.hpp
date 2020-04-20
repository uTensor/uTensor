#ifndef UTENSOR_QUANTIZE_OPS_H
#define UTENSOR_QUANTIZE_OPS_H
#include <algorithm>
#include <limits>

#include "operatorBase.hpp"

namespace uTensor {
namespace TFLM {
// https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/kernels/internal/reference/dequantize.h#L30-L44
template <typename OutputT, typename InputT>
void dequantize_kernel(Tensor& b, const Tensor& a) {
  
  // More extensive dimensional checks can be included here
  const int flat_size = b->get_shape().get_linear_size();
  const QuantizationParams& quantParam = a->get_quantization_params();
  int32_t zero_point = quantParam.get_zeroP_for_channel(0);
  float scale = quantParam.get_scale_for_channel(0);

  for (int i = 0; i < flat_size; i++) {
    const int32_t val = static_cast<InputT>(a(i));
    OutputT result = static_cast<OutputT>(scale * static_cast<float>((val - zero_point)));
    b(i) = result;
  }
}

template <typename oT, typename iT>
class DequantizeOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { a };
  enum names_out : uint8_t { b };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    dequantize_kernel<oT, iT>(outputs[b].tensor(), inputs[a].tensor());
  }
};

}  //namespace TFLM
}  // namespace uTensor

#endif
