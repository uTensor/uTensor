#ifndef UTENSOR_MATH_OPS
#define UTENSOR_MATH_OPS

#include <climits>
#include "quantization_utils.hpp"
#include "tensor.hpp"

void CalculateUsedRange(Tensor<int>& input, int32_t* used_min_quan,
                        int32_t* used_max_quan) {
  int32_t minimum = INT_MAX;
  int32_t maxmum = INT_MIN;
  uint32_t size = input.getSize();
  int* in_ptr = input.getPointer({});

  for (uint32_t i = 0; i < size; i++) {
    if (minimum > in_ptr[i]) minimum = static_cast<int32_t>(in_ptr[i]);

    if (maxmum < in_ptr[i]) maxmum = static_cast<int32_t>(in_ptr[i]);
  }

  *used_min_quan = minimum;
  *used_max_quan = maxmum;
}
template <class T1, class T2>
void Requantization_Range(Tensor<T1> input, Tensor<T2> min, Tensor<T2> max,
                          Tensor<T2> out_min, Tensor<T2> out_max) {
  const float input_min = *(min.getPointer({}));
  const float input_max = *(max.getPointer({}));

  int32_t used_min_quan;
  int32_t used_max_quan;
  CalculateUsedRange(input, &used_min_quan, &used_max_quan);

  const float used_min =
      std::min(0.0f, QuantizedToFloat(used_min_quan, input_min, input_max));
  const float used_max = QuantizedToFloat(used_max_quan, input_min, input_max);

  float* c_min = out_min.getPointer({});
  *c_min = used_min;
  float* c_max = out_max.getPointer({});
  *c_max = used_max;
}
template <class T1, class T2, class Toutput>
void Requantize(Tensor<T1> input, Tensor<T2> in_min, Tensor<T2> in_max,
                Tensor<T2> r_min, Tensor<T2> r_max, Tensor<Toutput> output,
                Tensor<T2> out_min, Tensor<T2> out_max) {
  const float input_min = in_min.getPointer({})[0];
  const float input_max = in_max.getPointer({})[0];
  const float r_output_min = r_min.getPointer({})[0];
  const float r_output_max = r_max.getPointer({})[0];

  RequantizeManyInNewRange<T1, Toutput>(input, input.getSize(), input_min,
                                        input_max, r_output_min, r_output_max,
                                        output);
  float* v_out_min = out_min.getPointer({});
  *v_out_min = r_output_min;
  float* v_out_max = out_max.getPointer({});
  *v_out_max = r_output_max;
}
template<class TIn, class TOut>
void Add(Tensor<TIn> input, Tensor<TIn> input2, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const TIn *p_in2 = input2.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = out.getSize();
  for (uint32_t i = 0; i < size; i++) {
    p_out[i] = p_in[i] + p_in2[i];
  }
}
template<class TIn, class Td, class TOut>
void Min(Tensor<TIn> input, Tensor<Td> dim, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const Td *p_in2 = dim.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = dim.getSize();
  vector<uint32_t> shape = input.getShape();
  TOut min = INT_MAX;
  for (uint32_t i = 0; i < size; i++) {
    Td n_dim = p_in2[i];
    uint32_t num = shape[n_dim];
    uint32_t stride = input.getStride(n_dim);
    for (uint32_t j = stride; j < stride + num; j++) {
        if (p_in[j] < min) {
            min = p_in[j];
        }
    }
    p_out[i] = min;
    min = INT_MAX;   
  }
}
template<class TIn, class Td, class TOut>
void Max(Tensor<TIn> input, Tensor<Td> dim, Tensor<TOut> out) {
  const TIn *p_in = input.getPointer({});
  const Td *p_in2 = dim.getPointer({});
  TOut *p_out = out.getPointer({});

  const uint32_t size = dim.getSize();
  vector<uint32_t> shape = input.getShape();
  TOut max = INT_MIN;
  for (uint32_t i = 0; i < size; i++) {
    Td n_dim = p_in2[i];
    uint32_t num = shape[n_dim];
    uint32_t stride = input.getStride(n_dim);
    for (uint32_t j = stride; j < stride + num; j++) {
        if (p_in[j] > max) {
            max = p_in[j];
        }
    }
    p_out[i] = max;
    max = INT_MIN;
  }
}

template<class TIn, class TOut>
void ArgMax(Tensor<TIn> input, Tensor<int> dim, Tensor<TOut> &out) {

  int dim_reduce = *(dim.getPointer({0}));
  Shape outShape = input.getShape();
  uint32_t reduce_dim_size = outShape[dim_reduce];
  outShape.erase(outShape.begin()+dim_reduce);

  //construct the permute vector
  vector<uint8_t> permute;
  for(uint8_t i = 0; i < input.getShape().size(); i++) {
    permute.push_back(i);
  }
  permute.push_back(static_cast<uint8_t>(dim_reduce));
  permute.erase(permute.begin()+dim_reduce);
  
  //check dimensionality
  if(out.getSize() != 0 && out.getShape() != outShape) {
    ERR_EXIT("output shape mismatch");
  }

  //allocate output tensor if empty
  if(out.getSize() == 0) {
    out = Tensor<TOut>(outShape);
  }

  //construct the origin-shape for permuteIndexTransform
  Shape vOutShape = outShape;
  vOutShape.push_back(reduce_dim_size);
  ///NT: easy way to remember...
  //trans(originShape, permute)
  //targetIndex = trans[OriginIndex]
  //In this case, we are going backward.
  permuteIndexTransform trans(vOutShape, permute);

  TIn* inPtr = input.getPointer({});
  TOut* outPtr = out.getPointer({});

  size_t out_index = 0;

  for(uint32_t i = 0; i < input.getSize(); i+=reduce_dim_size) {
    TOut max_j = 0;
    TIn last_max = std::numeric_limits<TIn>::min();
    for(uint32_t j = 0; j < reduce_dim_size; j++) {
      TIn val = inPtr[trans[i + j]];
      if(val > last_max) {
        last_max = val;
        max_j = j;
      }
    }

    outPtr[out_index] = max_j;
    out_index++;
  }
}
#endif  // UTENSOR_MATH_OPS
