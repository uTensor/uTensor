#ifndef UTENSOR_TANH_H
#define UTENSOR_TANH_H
#include <math.h>

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
    Tensor& out = inputs[act_out].tensor();

    const int flat_size = in->get_shape().get_linear_size();

    int32_t in_zero_point =
        in->get_quantization_params().get_zeroP_for_channel(0);
    float in_scale = in->get_quantization_params().get_scale_for_channel(0);

    int32_t out_zero_point =
        out->get_quantization_params().get_zeroP_for_channel(0);
    float out_scale = out->get_quantization_params().get_scale_for_channel(0);

    for (int i = 0; i < flat_size; i++) {
      const int32_t in_val = static_cast<InputT>(in(i));
      float float_in_val = static_cast<float>(
          in_scale * static_cast<float>((in_val - in_zero_point)));
      float element_activation = tanh(float_in_val);
      OutputT result =
          static_cast<OutputT>(element_activation / out_scale + out_zero_point);

      out(i) = result;
    }
  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor
#endif
