#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include <climits>
#include "quantization_utils.hpp"
#include "tensor.hpp"
#include "uTensorBase.hpp"

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
  if (output && output->getSize() == 0) {
  output->resize<Toutput>(input->getShape());
  }
  Toutput *out_ptr = output->write<Toutput>(0, 0);

  // RequantizeManyInNewRange<T1, Toutput>(input, input.getSize(), input_min,
  //                                       input_max, r_output_min, r_output_max,
  //                                       output);
  RequantizeManyInNewRangeReference(input_ptr, input->getSize(),input_min,
    input_max, r_output_min, r_output_max, out_ptr);

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
template <class TIn, class Td, class TOut>
void Min(S_TENSOR input, S_TENSOR dim, S_TENSOR out) {
  const TIn* p_in = input->read<TIn>(0, 0);
  const Td* p_in2 = dim->read<Td>(0, 0);
  TOut* p_out = out->write<TOut>(0, 0);

  Td n_dim = p_in2[0];
  std::vector<uint8_t> permute;
  for (uint32_t i_dim = 0; i_dim < input->getShape().size(); i_dim++) {
    permute.push_back(i_dim);
  }
  permute.push_back(n_dim);
  permute.erase(permute.begin() + n_dim);
  Shape outShape = input->getShape();
  size_t reduce_size = outShape[n_dim];
  outShape.erase(outShape.begin() + n_dim);
  outShape.push_back(reduce_size);
  size_t out_index = 0;
  permuteIndexTransform trans(outShape, permute);
  for (uint32_t j = 0; j < input->getSize(); j += reduce_size) {
    TIn min_val = std::numeric_limits<TIn>::max();
    for (size_t k = 0; k < reduce_size; k++) {
      TIn val = p_in[trans[j + k]];
      if (val < min_val) {
        min_val = val;
      }
    }
    p_out[out_index] = min_val;
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
      Min<float, int, float>(inputs[0], inputs[1], outputs[0]);
    }
};
template <class TIn, class Td, class TOut>
void Max(S_TENSOR input, S_TENSOR dim, S_TENSOR out) {
  const TIn* p_in = input->read<TIn>(0, 0);
  const Td* p_in2 = dim->read<Td>(0, 0);
  TOut* p_out = out->write<TOut>(0, 0);

  Td n_dim = p_in2[0];
  std::vector<uint8_t> permute;
  for (uint32_t i_dim = 0; i_dim < input->getShape().size(); i_dim++) {
    permute.push_back(i_dim);
  }
  permute.push_back(n_dim);
  permute.erase(permute.begin() + n_dim);
  Shape outShape = input->getShape();
  size_t reduce_size = outShape[n_dim];
  outShape.erase(outShape.begin() + n_dim);
  outShape.push_back(reduce_size);
  size_t out_index = 0;
  permuteIndexTransform trans(outShape, permute);
  for (uint32_t j = 0; j < input->getSize(); j += reduce_size) {
    TIn max_val = std::numeric_limits<TIn>::lowest();
    for (size_t k = 0; k < reduce_size; k++) {
      TIn val = p_in[trans[j + k]];
      if (val > max_val) {
        max_val = val;
      }
    }
    p_out[out_index] = max_val;
    out_index++;
  }
}

class MaxOp : public Operator {
  public:
    MaxOp() {
    n_inputs = 2;
    n_outputs = 1;
  }

  virtual void compute() override {
    Max<float, int, float>(inputs[0], inputs[1], outputs[0]);
  }
};

template <class TIn, class TOut>
void ArgMax(S_TENSOR input, S_TENSOR dim, S_TENSOR out) {
  int dim_reduce = *(dim->read<int>(0, 0));
  Shape outShape = input->getShape();
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
  if (out && out->getSize() != 0 && out->getShape() != outShape) {
    ERR_EXIT("output shape mismatch");
  }

  // allocate output tensor if empty
  if (out && out->getSize() == 0) {
    out->resize<TOut>(outShape);
  }
  

  // construct the origin-shape for permuteIndexTransform
  Shape vOutShape = outShape;
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
  const TIn* p_in = input->read<TIn>(0, 0);
  const TIn* p_in2 = input2->read<TIn>(0, 0);

  //auto shape
  out->resize<TOut>(input->getShape());

  TOut* p_out = out->write<TOut>(0, 0);

  const uint32_t size = out->getSize();
  for (uint32_t i = 0; i < size; i++) {
    p_out[i] = p_in[i] + p_in2[i];
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
#endif  // UTENSOR_MATH_OPS
