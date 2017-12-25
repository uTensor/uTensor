#ifndef UTENSOR_MATRIX_OPS
#define UTENSOR_MATRIX_OPS

#include <cmath>
#include <cstdlib>
#include <limits>
#include "quantization_utils.hpp"
#include "tensor.hpp"

// tensorflow/tensorflow/core/kernels/reference_gemm.h

template <class T1, class T2, class T3>
void ReferenceGemmuImpl(bool transpose_a, bool transpose_b, bool transpose_c,
                        size_t m, size_t n, size_t k, S_TENSOR a,
                        int32_t offset_a, size_t lda, S_TENSOR b, int offset_b,
                        size_t ldb, S_TENSOR c, int shift_c, int offset_c,
                        int mult_c, size_t ldc) {
  int a_i_stride = lda;
  int a_l_stride = 1;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  }

  int b_j_stride = 1;
  int b_l_stride = ldb;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  }

  int c_i_stride = ldc;
  int c_j_stride = 1;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  }

  const int32_t highest = static_cast<int32_t>(std::numeric_limits<T3>::max());
  const int32_t lowest = static_cast<int32_t>(std::numeric_limits<T3>::min());
  const int32_t rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  size_t i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const T1* a_data = a->read<T1>(a_index, 1);
        const int32_t a_value = static_cast<int32_t>(a_data[0]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const T2* b_data = b->read<T2>(b_index, 1);
        const int32_t b_value = static_cast<int32_t>(b_data[0]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      T3* c_data = c->write<T3>(c_index, 1);
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }

      if (output < lowest) {
        output = lowest;
      }
      c_data[0] = static_cast<T3>(output);
    }
  }
}

template <class T>
float FloatForOneQuantizedLevel(
    float range_min,
    float
        range_max)  // NT: information loss if float_for_one_quantized_level < 1
{
  const int64_t highest = static_cast<int64_t>(std::numeric_limits<T>::max());
  const int64_t lowest = static_cast<int64_t>(std::numeric_limits<T>::lowest());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64_t c_highest =
      static_cast<int64_t>(std::numeric_limits<T3>::max());
  const int64_t c_lowest =
      static_cast<int64_t>(std::numeric_limits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;  // NT: this resulting in
                                                    // taking only the necessary
                                                    // quantize range
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2, class Toutput>
void QuantizedMatMul(Tensor* A, Tensor* B, Tensor** C,
                     Tensor* mina, Tensor* minb, Tensor* maxa,
                     Tensor* maxb, Tensor* outmin,
                     Tensor* outmax, bool transpose_a = false,
                     bool transpose_b = false) {
  const float min_a = *(mina->read<float>(0, 0));
  const float max_a = *(maxa->read<float>(0, 0));
  const float min_b = *(minb->read<float>(0, 0));
  const float max_b = *(maxb->read<float>(0, 0));

  //auto tensor allocation
  Shape c_shape;
  c_shape.push_back((A->getShape())[0]);
  c_shape.push_back((B->getShape())[1]);
  tensorChkAlloc<Toutput>(C, c_shape);

  const int32_t offset_a = FloatToQuantizedUnclamped<T1>(
      0.0f, min_a, max_a);  // NT: what 0 quantized to; depends on
                            // Eigen::NumTraits<T>::lowest()
  const int32_t offset_b = FloatToQuantizedUnclamped<T2>(0.0f, min_b, max_b);
  const int32_t offset_c = 0;
  const int32_t mult_c = 1;
  const int32_t shift_c = 0;

  int first = transpose_a ? 0 : 1;
  int second = transpose_b ? 1 : 0;

  int a_dim_remaining = 1 - first;
  int b_dim_remaining = 1 - second;

  const T1* A_Data = A->read<T1>(0, 0);
  const T2* B_Data = B->read<T2>(0, 0);
  Toutput* C_Data = (*C)->write<Toutput>(0, 0);

  const bool transpose_c = false;
  const size_t m = A->getShape()[a_dim_remaining];
  const size_t n = B->getShape()[b_dim_remaining];
  const size_t k = A->getShape()[first];
  const size_t lda = A->getShape()[1];
  const size_t ldb = B->getShape()[1];
  const size_t ldc = n;

  ReferenceGemmuImpl<T1, T2, Toutput>(
      transpose_a, transpose_b, transpose_c, m, n, k, A_Data, offset_a, lda,
      B_Data, offset_b, ldb, C_Data, shift_c, offset_c, mult_c, ldc);
  float min_c_value;
  float max_c_value;

  QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);

  float* c_min = outmin->write<float>(0, 0);
  *c_min = min_c_value;
  float* c_max = outmax->write<float>(0, 0);
  *c_max = max_c_value;
}

//////////////////////////////////////////////////////
template <class T1, class T2, class Toutput>
void QuantizedMatMul2(S_TENSOR A, S_TENSOR B, S_TENSOR C,
                     S_TENSOR mina, S_TENSOR minb, S_TENSOR maxa,
                     S_TENSOR maxb, S_TENSOR outmin,
                     S_TENSOR outmax, bool transpose_a = false,
                     bool transpose_b = false) {
  const float min_a = *(mina->read<float>(0, 0));
  const float max_a = *(maxa->read<float>(0, 0));
  const float min_b = *(minb->read<float>(0, 0));
  const float max_b = *(maxb->read<float>(0, 0));

  //auto tensor allocation
  if(C->getSize() == 0) {
    Shape c_shape;
    c_shape.push_back((A->getShape())[0]);
    c_shape.push_back((B->getShape())[1]);
    C->resize(c_shape);
  }
  

  const int32_t offset_a = FloatToQuantizedUnclamped<T1>(
      0.0f, min_a, max_a);  // NT: what 0 quantized to; depends on
                            // Eigen::NumTraits<T>::lowest()
  const int32_t offset_b = FloatToQuantizedUnclamped<T2>(0.0f, min_b, max_b);
  const int32_t offset_c = 0;
  const int32_t mult_c = 1;
  const int32_t shift_c = 0;

  int first = transpose_a ? 0 : 1;
  int second = transpose_b ? 1 : 0;

  int a_dim_remaining = 1 - first;
  int b_dim_remaining = 1 - second;

  const bool transpose_c = false;
  const size_t m = A->getShape()[a_dim_remaining];
  const size_t n = B->getShape()[b_dim_remaining];
  const size_t k = A->getShape()[first];
  const size_t lda = A->getShape()[1];
  const size_t ldb = B->getShape()[1];
  const size_t ldc = n;

  ReferenceGemmuImpl<T1, T2, Toutput>(
      transpose_a, transpose_b, transpose_c, m, n, k, A, offset_a, lda,
      B, offset_b, ldb, C, shift_c, offset_c, mult_c, ldc);
  float min_c_value;
  float max_c_value;

  QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);

  float* c_min = outmin->write<float>(0, 0);
  *c_min = min_c_value;
  float* c_max = outmax->write<float>(0, 0);
  *c_max = max_c_value;
}

template <class T1, class T2, class TOut>
class QntMatMulOp : public Operator {
public:
  QntMatMulOp() {
    n_inputs = 6;
    n_outputs = 3;
  }
  virtual void compute() override {
    QuantizedMatMul2<T1, T2, TOut>(inputs[0], inputs[3],
     outputs[0], inputs[1], inputs[4], inputs[2], inputs[5],
      outputs[1], outputs[2]);
  }
};

#endif
