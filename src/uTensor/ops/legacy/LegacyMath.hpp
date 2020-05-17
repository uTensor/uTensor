#ifndef UTENSOR_LEGACY_MATRIX_OPS_H
#define UTENSOR_LEGACY_MATRIX_OPS_H

#include <limits>
#include <algorithm>
#include <cmath>
#include <climits>

#include "context.hpp"
#include "operatorBase.hpp"
#include "legacyQuantizationUtils.hpp"

namespace uTensor {
namespace legacy {

template<class T1=int32_t>
void CalculateUsedRange(const Tensor& input, int32_t* used_min_quan,
                        int32_t* used_max_quan) {
  int32_t minimum = INT_MAX;
  int32_t maxmum = INT_MIN;
  uint32_t size = input->get_shape().num_elems();
  //const T1* in_ptr = input->read<T1>(0, 0);

  for (uint32_t i = 0; i < size; i++) {
    if (minimum > static_cast<T1>(input(i))) minimum = static_cast<int32_t>(input(i));

    if (maxmum < static_cast<T1>(input(i))) maxmum = static_cast<int32_t>(input(i));
  }

  *used_min_quan = minimum;
  *used_max_quan = maxmum;
}
template <class T1, class T2>
void Requantization_Range(const Tensor& input, const float& imin, const float& imax,
                          float& out_min, float& out_max) {
  const float input_min = imin;
  const float input_max = imax;

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange<T1>(input, &used_min_quan, &used_max_quan);

  //TensorShape one_shape = {1};
  //if(out_min->getSize() == 0) out_min->resize(one_shape);
  //if(out_max->getSize() == 0) out_max->resize(one_shape);

  const float used_min =
      std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);

  out_min = used_min;
  out_max = used_max;
}

class Requantization_RangeOp : public OperatorInterface<1,0> {
  public:
    enum names_in : uint8_t { in };
    enum names_out : uint8_t { };
    Requantization_RangeOp(const float& i_min, const float& i_max, float& out_min, float& out_max) : i_min(i_min), i_max(i_max), out_min(out_min), out_max(out_max) {
    }

  protected:
    virtual void compute() override {
      Requantization_Range<int32_t, float>(inputs[in].tensor(), 
        i_min, i_max,
        out_min, out_max);
    }
  private:
    const float& i_min;
    const float& i_max;
    float& out_min; 
    float& out_max;

};

// Does not reshape output, output must be sized correctly
template <class T1, class T2, class Toutput>
void Requantize(const Tensor& input, const float& in_min, const float& in_max,
                const float& r_min, const float& r_max, Tensor& output,
                float& out_min, float& out_max) {
  const float input_min = in_min; //->read<T2>(0, 0)[0];
  const float input_max = in_max;//->read<T2>(0, 0)[0];
  const float r_output_min = r_min;//->read<T2>(0, 0)[0];
  const float r_output_max = r_max;//->read<T2>(0, 0)[0];
  //const T1 *input_ptr = input->read<T1>(0, 0);

  //if (output->getSize() == 0) output->resize(input->getShape());
  //Toutput *out_ptr = output->write<Toutput>(0, 0);

  RequantizeManyInNewRangeReference(input, input->get_shape().num_elems(), input_min,
    input_max, r_output_min, r_output_max, output);

 //   TensorShape one_shape = {1};
 //   if(out_min->getSize() == 0) out_min->resize(one_shape);
 //   if(out_max->getSize() == 0) out_max->resize(one_shape);

  out_min = r_output_min;
  out_max = r_output_max;
}

//template<class T1, class T2>
class RequantizeOp : public OperatorInterface<1,1> {
  public:
    enum names_in : uint8_t { input };
    enum names_out : uint8_t { output };
    RequantizeOp(
      const float& input_min,
      const float& input_max,
      const float& r_min,
      const float& r_max,
      float& output_min,
      float& output_max
        ) :
      input_min(input_min),
      input_max(input_max),
      r_min(r_min),
      r_max(r_max),
      output_min(output_min),
      output_max(output_max) {
    }

    virtual void compute() override {
        Requantize<int, float, unsigned char>(inputs[input].tensor(), input_min, 
                input_max, r_min, r_max,
                outputs[output].tensor(), output_min, output_max);
    }

  private:
    const float& input_min;
    const float& input_max;
    const float& r_min;
    const float& r_max;
    float& output_min;
    float& output_max;
};

// Note input_x should have >= the number of elements in input_y
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_add_op.cc
// T1: input quantize type
// T2: input dequantize type
// Toutput: output quantize type
// the base template
template <class T1, class T2, class Toutput>
void QuantizedAdd(const Tensor& input_x, const Tensor& input_y,
                  const float& min_x, const float& max_x,
                  const float& min_y, const float& max_y,
                  Tensor& output, float& out_min, float& out_max) {
  const uint32_t input_element_count = input_x->get_shape().num_elems();
  const uint32_t smaller_input_element_count = input_y->get_shape().num_elems();
  const float value_x_min = min_x;
  const float value_x_max = max_x;
  const float value_y_min = min_y;
  const float value_y_max = max_y;
  float value_out_min = std::min(value_x_min, value_y_min);
  float value_out_max = std::max(value_x_max, value_y_max);

  //Toutput* ptr_out_min = out_min->write<Toutput>(0, 0);
  //Toutput* ptr_out_max = out_max->write<Toutput>(0, 0);
  //*(ptr_out_min) = static_cast<Toutput>(value_out_min);
  //*(ptr_out_max) = static_cast<Toutput>(value_out_max);
  //out_min = static_cast<Toutput>(value_out_min);
  //out_max = static_cast<Toutput>(value_out_max);
  out_min = value_out_min;
  out_max = value_out_max;

  const size_t num_iterations = (input_element_count / smaller_input_element_count);
  //const T1* ptr_x = input_x->read<T1>(0, 0);
  //const T1* ptr_y = input_y->read<T1>(0, 0);

  //if (!output->getSize()) output->resize(input_x->getShape());

  //Toutput* ptr_out = output->write<Toutput>(0, 0);
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t offset = i * smaller_input_element_count;
    for (size_t c = 0; c < smaller_input_element_count; ++c) {
      T1 x = input_x(offset + c);
      T1 y = input_y(c);
      Toutput new_x = RequantizeInNewRange<T1, Toutput>(x, value_x_min, value_x_max, value_out_min, value_out_max);
      Toutput new_y = RequantizeInNewRange<T1, Toutput>(y, value_y_min, value_y_max, value_out_min, value_out_max);
      output(offset+c) = new_x + new_y;
    }
  }
}

template <>
void QuantizedAdd<uint8_t, uint8_t, int>(
  const Tensor& input_x, const Tensor& input_y,
  const float& min_x, const float& max_x,
  const float& min_y, const float& max_y,
  Tensor& output, float& out_min, float& out_max);

// Note input_x should have >= the number of elements in input_y
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_add_op.cc
// T1: input quantize type
// T2: input dequantize type
// Toutput: output quantize type
// the base template
template <class T1, class T2, class Toutput>
void QuantizedMul(const Tensor& input_x, const Tensor& input_y,
                  const float&  min_x, const float&  max_x,
                  const float&  min_y, const float&  max_y,
                  Tensor& output, float& out_min, float& out_max) {
  const uint32_t input_element_count = input_x->get_shape().num_elems();
  const uint32_t smaller_input_element_count = input_y->get_shape().num_elems();
  const float value_x_min = min_x;
  const float value_x_max = max_x;
  const float value_y_min = min_y;
  const float value_y_max = max_y;

  const int32_t offset_x = FloatToQuantizedUnclamped<T1>(0.0f, value_x_min, value_x_max);  // NT: what 0 quantized to; depends on // Eigen::NumTraits<T>::lowest()
  const int32_t offset_y = FloatToQuantizedUnclamped<T2>(0.0f, value_y_min, value_y_max);
  const int32_t shift_out = 0;

  const int32_t highest = static_cast<int32_t>(std::numeric_limits<Toutput>::max());
  const int32_t lowest = static_cast<int32_t>(std::numeric_limits<Toutput>::min());
  const int32_t rounding = (shift_out < 1) ? 0 : (1 << (shift_out - 1));

  float value_out_min;
  float value_out_max;

  //float* ptr_out_min = out_min->write<float>(0, 0);
  //float* ptr_out_max = out_max->write<float>(0, 0);

  const size_t num_iterations = (input_element_count / smaller_input_element_count);
  //const T1* ptr_x = input_x->read<T1>(0, 0);
  //const T1* ptr_y = input_y->read<T1>(0, 0);

  // if (!output->getSize()) 
  //output->resize(input_x->getShape());

  //Toutput* ptr_out = output->write<Toutput>(0, 0);
  int num_elems = input_x->get_shape().num_elems();
  if(smaller_input_element_count == 1){
    for(int i = 0; i < num_elems; i++){
      Toutput t = (static_cast<int32_t>(static_cast<T1>(input_x(i))) - offset_x) *
                (static_cast<int32_t>(static_cast<T1>(input_y(0))) - offset_y);
      output(i) = t;
    }
  } else { //Vector multiply
    for(int i = 0; i < smaller_input_element_count; i++){
      Toutput t = (static_cast<int32_t>(static_cast<T1>(input_x(i))) - offset_x) *
                (static_cast<int32_t>(static_cast<T1>(input_y(i))) - offset_y);
      output(i) = t;
    } //Tensor multiply as well???
  }

  QuantizationRangeForMultiplication<T1, T2, Toutput>(value_x_min, value_x_max, value_y_min, value_y_max,
      &value_out_min, &value_out_max);
  
  out_min = value_out_min;
  out_max = value_out_max;
  //*(ptr_out_min) = static_cast<float>(value_out_min);
  //*(ptr_out_max) = static_cast<float>(value_out_max);
}

/*
template <>
void QuantizedMul<uint8_t, uint8_t, int>(
  S_TENSOR input_x, S_TENSOR input_y,
  S_TENSOR min_x, S_TENSOR max_x,
  S_TENSOR min_y, S_TENSOR max_y,
  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max);
*/
template<class T1, class T2, class T3>
class QuantizedAddOp : public OperatorInterface<2,1> {
  public:
    enum names_in : uint8_t { a, b };
    enum names_out : uint8_t { c };
    QuantizedAddOp(
      const float& a_min,
      const float& a_max,
      const float& b_min,
      const float& b_max,
      float& c_min,
      float& c_max
        ) : a_min(a_min),
      a_max(a_max),
      b_min(b_min),
      b_max(b_max),
      c_min(c_min),
      c_max(c_max)
    {
    }

  protected:
    virtual void compute() override {
        QuantizedAdd<T1, T2, T3>(inputs[a].tensor(), inputs[b].tensor(),  
            a_min, a_max, b_min, b_max, 
            outputs[c].tensor(), c_min, c_max);
    }
  private:
    const float& a_min;
    const float& a_max;
    const float& b_min;
    const float& b_max;
    float& c_min;
    float& c_max;
};
template<class T1, class T2, class T3>
class QuantizedMulOp : public OperatorInterface<2,1> {
  public:
    enum names_in : uint8_t { a, b };
    enum names_out : uint8_t { c };
    QuantizedMulOp(
      const float& a_min,
      const float& a_max,
      const float& b_min,
      const float& b_max,
      float& c_min,
      float& c_max
        ) :
      a_min(a_min),
      a_max(a_max),
      b_min(b_min),
      b_max(b_max),
      c_min(c_min),
      c_max(c_max)
    {
    }

  protected:
    virtual void compute() override {
        QuantizedMul<T1, T2, T3>(inputs[a].tensor(), inputs[b].tensor(),  
            a_min, a_max, b_min, b_max, 
            outputs[c].tensor(), c_min, c_max);
    }
  private:
    const float& a_min;
    const float& a_max;
    const float& b_min;
    const float& b_max;
    float& c_min;
    float& c_max;
};

}
}
#endif
