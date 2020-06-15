#ifndef UTENSOR_QUANTIZE_OPS_H
#define UTENSOR_QUANTIZE_OPS_H
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "operatorBase.hpp"

namespace uTensor {
namespace TflmSymQuantOps {

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
    OutputT result =
        static_cast<OutputT>(scale * static_cast<float>((val - zero_point)));
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

template <typename Tout, typename Tin>
void affine_quantize_kernel(Tensor& output, const Tensor& input) {
  // op:
  // https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/micro/kernels/quantize.cc
  // kernel:
  // https://github.com/tensorflow/tensorflow/blob/fb4ec5cbde3973050e7350f0aca7f07ab7757bac/tensorflow/lite/kernels/internal/reference/quantize.h
  const QuantizationParams& quant_params = output->get_quantization_params();
  if (output->num_elems() == 0) {
    output->resize(input->get_shape());
  }
  if (input->num_elems() != output->num_elems()) {
    uTensor_printf(
        "number of elements of output tensor mismatch with the input for "
        "quantization\n");
    Context::get_default_context()->throwError(new InvalidTensorOutputError);
    return;
  }
  const int32_t zp = quant_params.get_zeroP_for_channel(0);
  const float scale = quant_params.get_scale_for_channel(0);
  const int32_t minVal = static_cast<int32_t>(std::numeric_limits<Tout>::min());
  const int32_t maxVal = static_cast<int32_t>(std::numeric_limits<Tout>::max());
  for (uint32_t i = 0; i < input->num_elems(); i++) {
    const Tin inVal = input(i);
    const float inVal_f = static_cast<float>(inVal);
    int32_t unclamped = static_cast<int32_t>(std::round(inVal_f / scale)) + zp;
    int32_t clamped = std::min(std::max(unclamped, minVal), maxVal);
    output(i) = static_cast<Tout>(clamped);
  }
}
// TODO @mbartling  Add template specializations for invalid type combos to
// sanity check
template <typename Tout, typename Tin>
class QuantizeOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };

 protected:
  void compute() {
    affine_quantize_kernel<Tout, Tin>(outputs[output].tensor(),
                                      inputs[input].tensor());
  }
};

}  // namespace TFLM
}  // namespace uTensor

#endif
