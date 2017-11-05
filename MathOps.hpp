#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include <climits>
#include "quantization_utils.hpp"
#include "tensor.hpp"

template<class T1>
void CalculateUsedRange(Tensor* input, int32_t* used_min_quan,
                        int32_t* used_max_quan) {
  int32_t minimum = INT_MAX;
  int32_t maxmum = INT_MIN;
  uint32_t size = input->getSize();
  T1* in_ptr = input->read<T1>(0, 0);

  for (uint32_t i = 0; i < size; i++) {
    if (minimum > in_ptr[i]) minimum = static_cast<int32_t>(in_ptr[i]);

    if (maxmum < in_ptr[i]) maxmum = static_cast<int32_t>(in_ptr[i]);
  }

  *used_min_quan = minimum;
  *used_max_quan = maxmum;
}
template <class T1, class T2>
void Requantization_Range(Tensor* input, Tensor* min, Tensor* max,
                          Tensor* out_min, Tensor* out_max) {
  const float input_min = *(min->read<T2>(0, 0));
  const float input_max = *(max->read<T2>(0, 0));

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange<T1>(input, &used_min_quan, &used_max_quan);

  const float used_min =
      std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);

  float* c_min = out_min->write<T2>(0, 0);
  *c_min = used_min;
  float* c_max = out_max->write<T2>(0, 0);
  *c_max = used_max;
}

template <class T1, class T2, class Toutput>
void Requantize(Tensor* input, Tensor* in_min, Tensor* in_max,
                Tensor* r_min, Tensor* r_max, Tensor* output,
                Tensor* out_min, Tensor* out_max) {
  const float input_min = in_min->read<T2>(0, 0)[0];
  const float input_max = in_max->read<T2>(0, 0)[0];
  const float r_output_min = r_min->read<T2>(0, 0)[0];
  const float r_output_max = r_max->read<T2>(0, 0)[0];
  T1 *input_ptr = input->read<T1>(0, 0);
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
void Min(Tensor* input, Tensor* dim, Tensor* out) {
  const TIn* p_in = input->read<TIn>(0, 0);
  const Td* p_in2 = dim->read<Td>(0, 0);
  TOut* p_out = out->read<TOut>(0, 0);

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

template <class TIn, class Td, class TOut>
void Max(Tensor* input, Tensor* dim, Tensor* out) {
  const TIn* p_in = input->read<TIn>(0, 0);
  const Td* p_in2 = dim->read<Td>(0, 0);
  TOut* p_out = out->read<TOut>(0, 0);

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

template <class TIn, class TOut>
void ArgMax(Tensor* input, Tensor* dim, Tensor** out) {
  int dim_reduce = *(dim->read<int>(0, 0));
  Shape outShape = input->getShape();
  uint32_t reduce_dim_size = outShape[dim_reduce];
  outShape.erase(outShape.begin() + dim_reduce);

  // construct the permute vector
  vector<uint8_t> permute;
  for (uint8_t i = 0; i < input->getShape().size(); i++) {
    permute.push_back(i);
  }
  permute.push_back(static_cast<uint8_t>(dim_reduce));
  permute.erase(permute.begin() + dim_reduce);

  // check dimensionality
  if (*out && (*out)->getSize() != 0 && (*out)->getShape() != outShape) {
    ERR_EXIT("output shape mismatch");
  }

  // allocate output tensor if empty
  if (*out && (*out)->getSize() == 0) {
    (*out)->init<TOut>(outShape);
  } else {
      *out = new RamTensor<TOut>(outShape);
  }
  

  // construct the origin-shape for permuteIndexTransform
  Shape vOutShape = outShape;
  vOutShape.push_back(reduce_dim_size);
  /// NT: easy way to remember...
  // trans(originShape, permute)
  // targetIndex = trans[OriginIndex]
  // In this case, we are going backward.
  permuteIndexTransform trans(vOutShape, permute);

  TIn* inPtr = input->read<TIn>(0, 0);
  TOut* outPtr = (*out)->write<TOut>(0, 0);

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
#endif  // UTENSOR_MATH_OPS
