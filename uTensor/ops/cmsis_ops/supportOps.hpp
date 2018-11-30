#ifndef UTENSOR_SUPPORT_OPS
#define UTENSOR_SUPPORT_OPS
#include "uTensor/core/tensor.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include "uTensor/uTensor/ops/MatrixOps.hpp"
#include "uTensor/util/quantization_utils.hpp"
#include "arm_math.h"
#include "arm_nnfunctions.h"

extern void uint8_to_q7_origin(const uint8_t *input, float &min, float &max, q7_t *out, size_t length);

template <typename TIn, typename TOut>
extern void right_shift(const TIn *input, TOut *out, const uint8_t shift, const size_t length);

class Uint8Q7OriginOp : public Operator {
  public:
  Uint8Q7OriginOp() {
    n_inputs = 3;
    n_outputs = 1;
  }
  virtual void compute() override {
    const uint8_t *input = inputs[0]->read<uint8_t>(0, 1);
    float min = *(inputs[1]->read<float>(0, 1));
    float max = *(inputs[2]->read<float>(0, 1));

    if(outputs[0]->getSize() == 0) {
      outputs[0]->resize(inputs[0]->getShape());
    }

    q7_t *out = outputs[0]->write<q7_t>(0, 1);

    uint8_to_q7_origin(input, min, max, out, inputs[0]->getSize());
    // printTensor<q7_t>(outputs[0], 16);

  }
};

template <class T1, class T2, class Toutput>
class QuantRangeForMultiplicationOp : public Operator {
  public:
  QuantRangeForMultiplicationOp() {
    n_inputs = 4;
    n_outputs = 2;
  }
  virtual void compute() override {
    float min_a = *(inputs[0]->read<float>(0, 1));
    float max_a = *(inputs[1]->read<float>(0, 1));
    float min_b = *(inputs[2]->read<float>(0, 1));
    float max_b = *(inputs[3]->read<float>(0, 1));

    if(outputs[0]->getSize() == 0) {
      Shape out_shape;
      out_shape.push_back(1);
      outputs[0]->resize(out_shape);
    }

    if(outputs[1]->getSize() == 0) {
      Shape out_shape;
      out_shape.push_back(1);
      outputs[1]->resize(out_shape);
    }

    // q7_t *out = outputs[0]->write<q7_t>(0, 1);  //unused variable?

    float *min_c_value = outputs[0]->write<float>(0, 1);
    float *max_c_value = outputs[1]->write<float>(0, 1);

    QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, min_c_value, max_c_value);
  }
};

#endif