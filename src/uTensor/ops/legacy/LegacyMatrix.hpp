#ifndef UTENSOR_LEGACY_MATRIX_OPS_H
#define UTENSOR_LEGACY_MATRIX_OPS_H

#include <limits>

#include "context.hpp"
#include "operatorBase.hpp"
#include "legacyQuantizationUtils.hpp"

namespace uTensor {
namespace legacy {

DECLARE_ERROR(InvalidLegacyMatrixMultIndicesError);

// tensorflow/tensorflow/core/kernels/reference_gemm.h
template <class T1, class T2, class T3>
void ReferenceGemmuImpl(bool transpose_a, bool transpose_b, bool transpose_c,
                        size_t m, size_t n, size_t k, const Tensor& a,
                        int32_t offset_a, size_t lda, const Tensor& b, int offset_b,
                        size_t ldb, Tensor& c, int shift_c, int offset_c,
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
        //const T1* a_data = a->read<T1>(a_index, 1);
        //const int32_t a_value = static_cast<int32_t>(a_data[0]) - offset_a;
        const int32_t a_value = static_cast<int32_t>(static_cast<T1>(a(a_index))) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        //const T2* b_data = b->read<T2>(b_index, 1);
        //const int32_t b_value = static_cast<int32_t>(b_data[0]) - offset_b;
        const int32_t b_value = static_cast<int32_t>(static_cast<T2>(b(b_index))) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      //T3* c_data = c->write<T3>(c_index, 1);
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }

      if (output < lowest) {
        output = lowest;
      }
      //c_data[0] = static_cast<T3>(output);
      c(c_index) = static_cast<T3>(output);
    }
  }
}

template <class T1, class T2, class Toutput>
void QuantizedMatMul(const Tensor& A, const Tensor& B, Tensor* C,
                     const float& mina, const float& minb,
                     const float& maxa, const float& maxb
                     float& outmin, float& outmax,
                     bool transpose_a = false,
                     bool transpose_b = false) {
  const float min_a = mina;
  const float max_a = maxa;
  const float min_b = minb;
  const float max_b = maxb;

  //auto tensor allocation
  TensorShape a_shape = A->get_shape();
  TensorShape b_shape = B->get_shape();
  TensorShape c_shape = C->get_shape();
  if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
      c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
      a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
    printf("[Error] Invalid matrix multiple shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidLegacyMatrixMultIndicesError);
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

  outmin = min_c_value;
  outmax = max_c_value;
}

template <class T1, class T2, class TOut>
class QntMatMulOp : public OperatorInterface<2,1> {
  public:
    enum names_in : uint8_t { a, b };
    enum names_out : uint8_t { c };

    QntMatMulOp(const float& a_min, const float& a_max, const float& b_min, const float& b_max, float& c_min, float& c_max) : a_min(a_min), a_max(a_max), b_min(b_min), b_max(b_max), c_min(c_min), c_max(c_max) {
  }

  protected:
  virtual void compute() override {
    QuantizedMatMul2<T1, T2, TOut>(
      inputs[a].tensor(),
      inputs[b].tensor(),
      outputs[c].tensor(),
      min_a, min_b,
      max_a, max_b,
      min_c, max_c);
  }
  private:
  // Quantization ranges
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
