#ifndef UTENSOR_TRIGONOMETRIC_H
#define UTENSOR_TRIGONOMETRIC_H
#include <cmath>

#include "uTensor/core/context.hpp"
#include "uTensor/core/operatorBase.hpp"
// #include "ActivationFncs.hpp"  //in-plcae interface

using std::sin;
using std::tanh;

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

    for (uint32_t i = 0; i < flat_size; i++) {
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

template <typename OutputT, typename InputT>
class SinOperator : public OperatorInterface<1, 1> {
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

    for (uint32_t i = 0; i < flat_size; i++) {
      const int32_t in_val = static_cast<InputT>(in(i));
      float float_in_val = static_cast<float>(
          in_scale * static_cast<float>((in_val - in_zero_point)));
      float element_activation = sin(float_in_val);
      OutputT result =
          static_cast<OutputT>(element_activation / out_scale + out_zero_point);
      out(i) = result;
    }
  }
};

template <>
class TanhOperator<float, float> : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { act_in };
  enum names_out : uint8_t { act_out };

 protected:
  virtual void compute();
};

template <>
class SinOperator<float, float> : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { act_in };
  enum names_out : uint8_t { act_out };

 protected:
  virtual void compute();
};

}  // namespace ReferenceOperators
}  // namespace uTensor
#endif  // UTENSOR_TRIGONOMETRIC_H
