#ifndef UTENSOR_TANH_H
#define UTENSOR_TANH_H

#include "context.hpp"
#include "operatorBase.hpp"
// #include "ActivationFncs.hpp"  //in-plcae interface

namespace uTensor {
namespace ReferenceOperators {

// adapt to tanh_kernel

template <typename OutputT, typename InputT>
class TanhOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { act_in };
  enum names_out : uint8_t { act_out };

 protected:
  virtual void compute() {
    Tensor& in = inputs[act_in].tensor();
    Tensor& out = outputs[act_out].tensor();

    const uint32_t flat_size = in->get_shape().get_linear_size();

    int32_t in_zero_point =
        in->get_quantization_params().get_zeroP_for_channel(0);
    float in_scale = in->get_quantization_params().get_scale_for_channel(0);

    int32_t out_zero_point =
        out->get_quantization_params().get_zeroP_for_channel(0);
    float out_scale = out->get_quantization_params().get_scale_for_channel(0);

  // Insert your code here
  // Iterate through elements in the input tensor
  // Convert each element to from quantized fix-point to float
  // Apply tanh to each element
  // Quantize the elements and write them to the output tensor

  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor
#endif
