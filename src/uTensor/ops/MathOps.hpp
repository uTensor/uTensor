#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include "src/uTensor/util/quantization_utils.hpp"
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include <climits>
#include <algorithm>
#include <vector>
#include <cmath>

template<class T1>
void CalculateUsedRange(Tensor* input, int32_t* used_min_quan,
                        int32_t* used_max_quan) {
  int32_t minimum = INT_MAX;
  int32_t maxmum = INT_MIN;
  uint32_t size = input->getSize();
  const T1* in_ptr = input->read<T1>(0, 0);

  for (uint32_t i = 0; i < size; i++) {
    if (minimum > in_ptr[i]) minimum = static_cast<int32_t>(in_ptr[i]);

    if (maxmum < in_ptr[i]) maxmum = static_cast<int32_t>(in_ptr[i]);
  }

  *used_min_quan = minimum;
  *used_max_quan = maxmum;
}
template <class T1, class T2>
void Requantization_Range(S_TENSOR input, S_TENSOR min, S_TENSOR max,
                          S_TENSOR out_min, S_TENSOR out_max) {
  const float input_min = *(min->read<T2>(0, 0));
  const float input_max = *(max->read<T2>(0, 0));

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange<T1>(input.get(), &used_min_quan, &used_max_quan);

  TensorShape one_shape = {1};
  if(out_min->getSize() == 0) out_min->resize(one_shape);
  if(out_max->getSize() == 0) out_max->resize(one_shape);

  const float used_min =
      std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);

  float* c_min = out_min->write<T2>(0, 0);
  *c_min = used_min;
  float* c_max = out_max->write<T2>(0, 0);
  *c_max = used_max;
}

class Requantization_RangeOp : public Operator {
  public:
    Requantization_RangeOp() {
      n_inputs = 3;
      n_outputs = 2;
    }

    virtual void compute() override {
      Requantization_Range<int, float>(inputs[0], inputs[1],
              inputs[2], outputs[0], outputs[1]);
    }
};
template <class T1, class T2, class Toutput>
void Requantize(S_TENSOR input, S_TENSOR in_min, S_TENSOR in_max,
                S_TENSOR r_min, S_TENSOR r_max, S_TENSOR output,
                S_TENSOR out_min, S_TENSOR out_max) {
  const float input_min = in_min->read<T2>(0, 0)[0];
  const float input_max = in_max->read<T2>(0, 0)[0];
  const float r_output_min = r_min->read<T2>(0, 0)[0];
  const float r_output_max = r_max->read<T2>(0, 0)[0];
  const T1 *input_ptr = input->read<T1>(0, 0);

  if (output->getSize() == 0) output->resize(input->getShape());
  Toutput *out_ptr = output->write<Toutput>(0, 0);

  // RequantizeManyInNewRange<T1, Toutput>(input, input.getSize(), input_min,
  //                                       input_max, r_output_min, r_output_max,
  //                                       output);
  RequantizeManyInNewRangeReference(input_ptr, input->getSize(),input_min,
    input_max, r_output_min, r_output_max, out_ptr);

    TensorShape one_shape = {1};
    if(out_min->getSize() == 0) out_min->resize(one_shape);
    if(out_max->getSize() == 0) out_max->resize(one_shape);

  float* v_out_min = out_min->write<T2>(0, 0);
  *v_out_min = r_output_min;
  float* v_out_max = out_max->write<T2>(0, 0);
  *v_out_max = r_output_max;
}


class RequantizeOp : public Operator {
  public:
    RequantizeOp() {
      n_inputs = 5;
      n_outputs = 3;
    }

    virtual void compute() override {
        Requantize<int, float, unsigned char>(inputs[0], inputs[1], 
                inputs[2], inputs[3], inputs[4],
                outputs[0], outputs[1], outputs[2]);
    }
};

template <class TIn, class TOut>
void Add(Tensor* input, Tensor* input2, Tensor** out) {
  const TIn* p_in = input->read<TIn>(0, 0);
  const TIn* p_in2 = input2->read<TIn>(0, 0);

  //auto shape
  tensorChkAlloc<TOut>(out, input->getShape());

  TOut* p_out = (*out)->write<TOut>(0, 0);

  const uint32_t size = (*out)->getSize();
  for (uint32_t i = 0; i < size; i++) {
    p_out[i] = p_in[i] + p_in2[i];
  }
}

//reduce_shape actual output shape without the reduce dim
//out_shape intermediate shape with reduce dim in the last orders
inline void reduceShapeHelper(TensorShape input, TensorShape dim, TensorShape &reduce_shape, TensorShape &out_shape, std::vector<uint8_t> &perm, size_t &reduce_size) {
  reduce_shape.empty();
  out_shape.empty();
  perm.empty();

  for(auto i = 0; i < (int) input.size(); i++) {
    if(std::find(dim.begin(), dim.end(), i) == dim.end()) {
      perm.push_back(i);
      out_shape.push_back(input[i]);
    }
  }

  reduce_shape = out_shape;
  reduce_size = 0;

  for(auto d:dim) {
    perm.push_back(d);
    out_shape.push_back(input[d]);
    if(reduce_size == 0) {
      reduce_size = input[d];
    } else {
      reduce_size *= input[d];
    }
  }
}

template <class TIn, class TOut>
inline TensorShape tensorToLinearVec(S_TENSOR input, S_TENSOR dim) {
  TensorShape vec;
  const TIn* ptr = dim->read<TIn>(0, 0);
  for(auto i = 0; i < (int) dim->getSize(); i++) {
    TIn curr_dim = ptr[i];
    if(curr_dim < 0) {
      curr_dim = input->getShape().size() + curr_dim;
    }

    vec.push_back(static_cast<TOut>(curr_dim));
  }

  return vec;
}



template <class TIn, class Td, class TOut>
void MinMaxHelper(S_TENSOR input, S_TENSOR dim, S_TENSOR out, bool find_min) {
  const TIn* p_in = input->read<TIn>(0, 0);
  TensorShape dim_vec = tensorToLinearVec<Td, uint32_t>(input, dim);

  TensorShape outShape;
  std::vector<uint8_t> permute;
  size_t reduce_size;
  TensorShape reduce_shape;
  reduceShapeHelper(input->getShape(), dim_vec, reduce_shape, outShape, permute, reduce_size);
  TensorShape one_shape = {1};
  if(out->getSize() == 0) out->resize(reduce_shape);  //TODO: dimension check here
  TOut* p_out = out->write<TOut>(0, 0);

  size_t out_index = 0;
  permuteIndexTransform trans(outShape, permute);
  for (uint32_t j = 0; j < input->getSize(); j += reduce_size) {

    TIn tmp_val;
    if(find_min) {
      tmp_val = std::numeric_limits<TIn>::max();
    } else {
      tmp_val = std::numeric_limits<TIn>::min();
    }

    for (size_t k = 0; k < reduce_size; k++) {
      TIn val = p_in[trans[j + k]];
      if(find_min) {
        if (val < tmp_val) {
          tmp_val = val;
        }
      } else {
        if (val > tmp_val) {
          tmp_val = val;
        }
      }
    }
    p_out[out_index] = tmp_val;
    out_index++;
  }
}

class MinOp : public Operator {
  public:
    MinOp() {
        n_inputs = 2;
        n_outputs = 1;
    }

    virtual void compute() override {
      MinMaxHelper<float, int, float>(inputs[0], inputs[1], outputs[0], true);
    }
};

class MaxOp : public Operator {
  public:
    MaxOp() {
    n_inputs = 2;
    n_outputs = 1;
  }

  virtual void compute() override {
    MinMaxHelper<float, int, float>(inputs[0], inputs[1], outputs[0], false);
  }
};

template <class TIn, class TOut>
void ArgMax(S_TENSOR input, S_TENSOR dim, S_TENSOR out) {
  int dim_reduce = *(dim->read<int>(0, 0));
  TensorShape outShape = input->getShape();
  uint32_t reduce_dim_size = outShape[dim_reduce];
  outShape.erase(outShape.begin() + dim_reduce);

  // construct the permute vector
  std::vector<uint8_t> permute;
  for (uint8_t i = 0; i < input->getShape().size(); i++) {
    permute.push_back(i);
  }
  permute.push_back(static_cast<uint8_t>(dim_reduce));
  permute.erase(permute.begin() + dim_reduce);

  // check dimensionality
  if (out->getSize() != 0 && out->getShape() != outShape) {
    ERR_EXIT("output shape mismatch");
  }

  // allocate output tensor if empty
  if (out->getSize() == 0) {
    out->resize(outShape);
  }
  

  // construct the origin-shape for permuteIndexTransform
  TensorShape vOutShape = outShape;
  vOutShape.push_back(reduce_dim_size);
  /// NT: easy way to remember...
  // trans(originShape, permute)
  // targetIndex = trans[OriginIndex]
  // In this case, we are going backward.
  permuteIndexTransform trans(vOutShape, permute);

  const TIn* inPtr = input->read<TIn>(0, 0);
  TOut* outPtr = out->write<TOut>(0, 0);

  size_t out_index = 0;

  for (uint32_t i = 0; i < input->getSize(); i += reduce_dim_size) {
    TOut max_j = 0;
    TIn last_max = std::numeric_limits<TIn>::min();
    for (uint32_t j = 0; j < reduce_dim_size; j++) {
      TIn val = inPtr[trans[i + j]];
      if (val > last_max) {
        last_max = val;
        max_j = j;
      }
    }

    outPtr[out_index] = max_j;
    out_index++;
  }
}


template<class TIn, class TOut>
class ArgMaxOp : public Operator {
  public:
    ArgMaxOp() {
      n_inputs = 2;
      n_outputs = 1;
  }

  virtual void compute() override {
    ArgMax<TIn, TOut>(inputs[0], inputs[1], outputs[0]);
  }
};

template <class TIn, class TOut>
void Add(S_TENSOR input, S_TENSOR input2, S_TENSOR out) {

  broadcastIndexTransform transf(input->getShape(), input2->getShape());
  S_TENSOR tmp_sptr;
  if(transf.is_swaped()) {
    tmp_sptr = input2;
    input2 = input;
    input = tmp_sptr;
  }

  const TIn* p_in = input->read<TIn>(0, 0);
  const TIn* p_in2 = input2->read<TIn>(0, 0);

  //auto shape
  out->resize(transf.getOutputShape());

  TOut* p_out = out->write<TOut>(0, 0);

  const uint32_t size = out->getSize();
  for (uint32_t i = 0; i < size; i++) {
    p_out[i] = p_in[i] + p_in2[transf[i]];
  }
}

template<class T1, class T2>
class AddOp : public Operator{
public:
  AddOp() {
    n_inputs = 2;
    n_outputs = 1;
  }
  virtual void compute() override {
    Add<T1, T2>(inputs[0], inputs[1], outputs[0]);
  }
};

// Note input_x should have >= the number of elements in input_y
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_add_op.cc
// T1: input quantize type
// T2: input dequantize type
// Toutput: output quantize type
// the base template
template <class T1, class T2, class Toutput>
void QuantizedAdd(S_TENSOR input_x, S_TENSOR input_y,
                  S_TENSOR min_x, S_TENSOR max_x,
                  S_TENSOR min_y, S_TENSOR max_y,
                  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {
  const uint32_t input_element_count = input_x->getSize();
  const uint32_t smaller_input_element_count = input_y->getSize();
  const float value_x_min = *(min_x->read<T2>(0, 0));
  const float value_x_max = *(max_x->read<T2>(0, 0));
  const float value_y_min = *(min_y->read<T2>(0, 0));
  const float value_y_max = *(max_y->read<T2>(0, 0));
  float value_out_min = std::min(value_x_min, value_y_min);
  float value_out_max = std::max(value_x_max, value_y_max);

  Toutput* ptr_out_min = out_min->write<Toutput>(0, 0);
  Toutput* ptr_out_max = out_max->write<Toutput>(0, 0);
  *(ptr_out_min) = static_cast<Toutput>(value_out_min);
  *(ptr_out_max) = static_cast<Toutput>(value_out_max);

  const size_t num_iterations = (input_element_count / smaller_input_element_count);
  const T1* ptr_x = input_x->read<T1>(0, 0);
  const T1* ptr_y = input_y->read<T1>(0, 0);

  if (!output->getSize()) output->resize(input_x->getShape());
  Toutput* ptr_out = output->write<Toutput>(0, 0);
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t offset = i * smaller_input_element_count;
    for (size_t c = 0; c < smaller_input_element_count; ++c) {
      T1 x = *(ptr_x + offset + c);
      T1 y = *(ptr_y + c);
      Toutput new_x = RequantizeInNewRange<T1, Toutput>(x, value_x_min, value_x_max, value_out_min, value_out_max);
      Toutput new_y = RequantizeInNewRange<T1, Toutput>(y, value_y_min, value_y_max, value_out_min, value_out_max);
      *(ptr_out+offset+c) = new_x + new_y;
    }
  }
}


// Note input_x should have >= the number of elements in input_y
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/quantized_add_op.cc
// T1: input quantize type
// T2: input dequantize type
// Toutput: output quantize type
// the base template
template <class T1, class T2, class Toutput>
void QuantizedMul(S_TENSOR input_x, S_TENSOR input_y,
                  S_TENSOR min_x, S_TENSOR max_x,
                  S_TENSOR min_y, S_TENSOR max_y,
                  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {
  const uint32_t input_element_count = input_x->getSize();
  const uint32_t smaller_input_element_count = input_y->getSize();
  const float value_x_min = *(min_x->read<float>(0, 0));
  const float value_x_max = *(max_x->read<float>(0, 0));
  const float value_y_min = *(min_y->read<float>(0, 0));
  const float value_y_max = *(max_y->read<float>(0, 0));

  const int32_t offset_x = FloatToQuantizedUnclamped<T1>(0.0f, value_x_min, value_x_max);  // NT: what 0 quantized to; depends on // Eigen::NumTraits<T>::lowest()
  const int32_t offset_y = FloatToQuantizedUnclamped<T2>(0.0f, value_y_min, value_y_max);
  const int32_t shift_out = 0;

  const int32_t highest = static_cast<int32_t>(std::numeric_limits<Toutput>::max());
  const int32_t lowest = static_cast<int32_t>(std::numeric_limits<Toutput>::min());
  const int32_t rounding = (shift_out < 1) ? 0 : (1 << (shift_out - 1));

  float value_out_min;
  float value_out_max;

  float* ptr_out_min = out_min->write<float>(0, 0);
  float* ptr_out_max = out_max->write<float>(0, 0);

  const size_t num_iterations = (input_element_count / smaller_input_element_count);
  const T1* ptr_x = input_x->read<T1>(0, 0);
  const T1* ptr_y = input_y->read<T1>(0, 0);

  // if (!output->getSize()) 
    output->resize(input_x->getShape());

  Toutput* ptr_out = output->write<Toutput>(0, 0);
  if(smaller_input_element_count == 1){
    for(int i = 0; i < input_x->getSize(); i++){
      Toutput t = (static_cast<int32_t>(ptr_x[i]) - offset_x) *
                (static_cast<int32_t>(ptr_y[0]) - offset_y);
      ptr_out[i] = t;
    }
  } else { //Vector multiply
    for(int i = 0; i < smaller_input_element_count; i++){
      Toutput t = (static_cast<int32_t>(ptr_x[i]) - offset_x) *
                (static_cast<int32_t>(ptr_y[i]) - offset_y);
      ptr_out[i] = t;
    } //Tensor multiply as well???
  }

  QuantizationRangeForMultiplication<T1, T2, Toutput>(value_x_min, value_x_max, value_y_min, value_y_max,
      &value_out_min, &value_out_max);
  
  *(ptr_out_min) = static_cast<float>(value_out_min);
  *(ptr_out_max) = static_cast<float>(value_out_max);
}

template <>
void QuantizedAdd<uint8_t, uint8_t, int>(
  S_TENSOR input_x, S_TENSOR input_y,
  S_TENSOR min_x, S_TENSOR max_x,
  S_TENSOR min_y, S_TENSOR max_y,
  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max);

/*
template <>
void QuantizedMul<uint8_t, uint8_t, int>(
  S_TENSOR input_x, S_TENSOR input_y,
  S_TENSOR min_x, S_TENSOR max_x,
  S_TENSOR min_y, S_TENSOR max_y,
  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max);
*/
template<class T1, class T2, class T3>
class QuantizedAddOp : public Operator {
  public:
      QuantizedAddOp() {
      n_inputs = 6;
      n_outputs = 3;
    }

    virtual void compute() override {
        QuantizedAdd<T1, T2, T3>(inputs[0], inputs[3],  
            inputs[1], inputs[2], inputs[4], inputs[5],
            outputs[0], outputs[1], outputs[2]);
    }
};

template<class T1, class T2, class T3>
class QuantizedMulOp : public Operator {
  public:
      QuantizedMulOp() {
      n_inputs = 6;
      n_outputs = 3;
    }

    virtual void compute() override {
        QuantizedMul<T1, T2, T3>(inputs[0], inputs[3],  
            inputs[1], inputs[2], inputs[4], inputs[5],
            outputs[0], outputs[1], outputs[2]);
    }
};


#endif  // UTENSOR_MATH_OPS
