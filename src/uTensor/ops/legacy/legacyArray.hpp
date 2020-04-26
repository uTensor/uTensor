#ifndef UTENSOR_LEGACY_ARRAY_OPS_H
#define UTENSOR_LEGACY_ARRAY_OPS_H

#include <limits>
#include <algorithm>
#include <cmath>

#include "context.hpp"
#include "operatorBase.hpp"
#include "legacyQuantizationUtils.hpp"

namespace uTensor {
namespace legacy {

DECLARE_ERROR(InvalidQuantizeV2RangeError);
  //T = inferred
//mode = MIN_FIRST
//name = unspecified
template <typename T>
void QuantizeV2(const Tensor& input, const float& _min_range, const float& _max_range,
                Tensor& output, float& output_min, float& output_max) {

    float input_min_range = _min_range;
    float input_max_range = _max_range;

    if(input_max_range < input_min_range) {
      //ERR_EXIT("input_max_range must be larger than input_min_range.");
      Context::get_default_context()->throwError(new InvalidQuantizeV2RangeError);
    }

    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                   fabsf(input_max_range))) / 100.0f;
    /*
    TensorShape v;

    TensorShape org = input->getShape();
    for (size_t i = 0; i < org.size(); i++) {
        v.push_back(org[i]);
    }

    if(output && output->getSize() == 0) {
      output->resize(v);
    }
    */

    float max_range = std::max(input_max_range, min_range + epsilon);
    max_range = std::max(0.0f, max_range);

    FloatToQuantizedStruct<T> f2q(min_range, max_range);

    //quantization_utils.h:149
    //const float* input_ptr = input->read<float>(0, 0);
    //T* output_ptr = output->write<T>(0, 0);
    //float* output_min_ptr = output_min->write<float>(0, 0);
    //float* output_max_ptr = output_max->write<float>(0, 0);

    ///NT: need error checking at some point...
    uint32_t num_elems = input->get_shape().num_elems();
    for(uint32_t i = 0; i < num_elems; i++) {
        float val = ::round(static_cast<float>(input_ptr(i)) * f2q.range_scale);
        val -= f2q.range_min_scaled - f2q.lowest_quantized();
        val = std::max(val, f2q.lower_bound_float());
        val = std::min(val, f2q.upper_bound_float());
        uint32_t intTmp = static_cast<uint32_t>(val); ///NT: omit this?
        output(i) = static_cast<T>(intTmp);
    }

    output_min = min_range;
    output_max = max_range;
    
}

class QuantizeV2Op : public OperatorInterface<1,1> {
  public:
    enum names_in : uint8_t { input };
    enum names_out : uint8_t { output };
    QuantizeV2Op(
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
      QuantizeV2<unsigned char>(inputs[input].tensor(), input_min, input_max,
              outputs[output].tensor(), output_min, output_max);
    }
  private:
    const float& input_min;
    const float& input_max;
    float& output_min;
    float& output_max;
}; 

//mode = MIN_FIRST
//name = unspecified
//dequantize_op.cc: 87
template <typename T>
void dequantize(const Tensor& input, const float& min_range, const float& max_range, Tensor& output) {
    float min = min_range;
    float max = max_range;
      //auto tensor allocation
    //TensorShape out_shape;
    //output->resize(input->getShape());

    //const T* input_ptr = input->read<T>(0, 0);
    //float* output_ptr = output->write<float>(0, 0);

    //quantization_utils.h: 771
    QuantizedToFloatStruct<T> q2f(min, max);

    //quantization_utils.h: 141
    uint32_t num_elems = input->get_shape().num_elems();
    for(uint32_t i = 0; i < num_elems; i++) {
        float val = static_cast<float>(static_cast<T>(input(i)));
        output(i) = ((q2f.range_min_rounded - q2f.lowest_quantized() * q2f.range_scale) + \
                        val * q2f.range_scale);
    }
}

class DequantizeOp : public OperatorInterface<1,1> {
  public:
    enum names_in : uint8_t { input };
    enum names_out : uint8_t { output };
    DequantizeOp(
      const float& input_min,
      const float& input_max) : input_min(input_min), input_max(input_max)
    {
    }

  protected:
    virtual void compute() override {
      dequantize<unsigned char>(inputs[input].tensor(), input_min, input_max,
              outputs[output].tensor());
    }
  private:
    const float& input_min;
    const float& input_max;
};

}
}

#endif
