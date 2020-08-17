#ifndef UTENSOR_MATRIX_OPS_H
#define UTENSOR_MATRIX_OPS_H
#include "context.hpp"
#include "operatorBase.hpp"
#include <math.h>
// #include "ActivationFncs.hpp"  //in-plcae interface

namespace uTensor {
namespace ReferenceOperators {

// adapt to tanh_kernel
}

template <typename OutputT, typename InputT>
class TanhOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { act_in };
  enum names_out : uint8_t { act_out };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() 
    Tensor& in = inputs[act_in].tensor()
    Tensor& out = inputs[act_out].tensor()

    const int flat_size = b->get_shape().get_linear_size();

    int32_t in_zero_point = in->get_quantization_params().get_zeroP_for_channel(0);
    float in_scale = in->get_quantization_params().get_scale_for_channel(0);

    int32_t out_zero_point = out->get_quantization_params().get_zeroP_for_channel(0);
    float out_scale = out->get_quantization_params().get_scale_for_channel(0);


    for (int i = 0; i < flat_size; i++) {
      const int32_t val = static_cast<InputT>(in(i));
      float float_val = static_cast<float>(in_scale * static_cast<float>((val - in_zero_point)));
      float element_activation = tanh(float_val);
      OutputT result = (element_activation / out_scale) + out_zero_point;

      out(i) = result;
    }

  }
};

}  // Reference Op
}  // namespace uTensor
#endif