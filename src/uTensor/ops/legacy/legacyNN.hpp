#ifndef UTENSOR_LEGACY_NN_OPS_H
#define UTENSOR_LEGACY_NN_OPS_H

#include <limits>
#include <algorithm>
#include <cmath>

#include "context.hpp"
#include "operatorBase.hpp"
#include "legacyQuantizationUtils.hpp"

namespace uTensor {
namespace legacy {

template <class TIn, class TOut>
void QuantizedRelu(const Tensor& input, const float& in_min, const float& in_max,
          Tensor& output, float& out_min, float& out_max) {
  const float input_min = in_min;
  const float input_max = in_max;
  //const TIn* in = input->read<TIn>(0, 0);

  const TOut min_as_quantized =
      FloatToQuantized<TOut>(0.0f, input_min, input_max);
  //if (output && output->getSize() == 0) {
  //    output->resize(input->getShape());
  //}
  //TOut* out = output->write<TOut>(0, 0);
  uint32_t num_elems = input->get_shape().num_elems();
  for (uint32_t i = 0; i < num_elems; i++) {
    if (static_cast<TIn>(input(i)) > min_as_quantized) {
      output(i) = static_cast<TOut>(static_cast<TIn>(input(i)));
    } else {
      output(i) = min_as_quantized;
    }
  }
  //T2* v_out_min = out_min->write<T2>(0, 0);
  //*v_out_min = input_min;
  //T2* v_out_max = out_max->write<T2>(0, 0);
  //*v_out_max = input_max;
  out_min = input_min;
  out_max = input_max;
}

template<class T1, class T2, class TOut>
class QuantizedReluOp : public OperatorInterface {
  public:
    enum names_in : uint8_t { input };
    enum names_out : uint8_t { output };
  
    QuantizedReluOp (
      const float& input_min,
      const float& input_max,
      float& output_min,
      float& output_max,
        ) :
      input_min(input_min),
      input_max(input_max),
      output_min(output_min),
      output_max(output_max),
    {
    }
  protected:
    virtual void compute() override {
      QuantizedRelu<T1, TOut>(inputs[input].tensor(), input_min, input_max, outputs[output].tensor(), output_min, output_max);
    }
  private:
    const float& input_min;
    const float& input_max;
    float& output_min;
    float& output_max;
};

}
}
#endif
