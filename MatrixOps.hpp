#ifndef UTENSOR_MATRIX_OPS
#define UTENSOR_MATRIX_OPS

#include <limits>
#include "test.hpp"

// Useful links
// http://www.plantation-productions.com/Webster/www.artofasm.com/Linux/HTML/Arraysa2.html

// Assume Tensor index order:  rows, columns | the first 2 dimensions
// typedef Tensor<unsigned char> Mat;

// void printMat(Mat mat) {
//     const uint32_t ROWS = mat.getShape()[0];
//     const uint32_t COLS = mat.getShape()[1];
//     unsigned char* pData = mat.getPointer({});

//     for(int i_r = 0; i_r < ROWS; i_r++) {
//         for(int i_c = 0; i_c < COLS; i_c++) {
//             printf("%f ", pData[i_r * COLS + i_c]);
//         }
//         printf("\r\n");
//     }
// }

// void initMatConst(Mat mat, unsigned char value) {
//     const uint32_t ROWS = mat.getShape()[0];
//     const uint32_t COLS = mat.getShape()[1];
//     unsigned char* pData = mat.getPointer({});

//     for(int i_r = 0; i_r < ROWS; i_r++) {
//         for(int i_c = 0; i_c < COLS; i_c++) {
//             pData[i_r * COLS + i_c] = value;
//         }
//     }
// }

// void multMat(Mat A, Mat B, Mat C) {
//     const uint32_t A_ROWS = A.getShape()[0];
//     const uint32_t A_COLS = A.getShape()[1];
//     const uint32_t B_ROWS = B.getShape()[0];
//     const uint32_t B_COLS = B.getShape()[1];
//     const uint32_t C_ROWS = C.getShape()[0];
//     const uint32_t C_COLS = C.getShape()[1];
//     unsigned char* A_Data = A.getPointer({});
//     unsigned char* B_Data = B.getPointer({});
//     unsigned char* C_Data = C.getPointer({});

//     if(A_COLS != B_ROWS) {
//         printf("A and B matrices dimension mismatch\r\n");
//         return;
//     }

//     if(C_ROWS != A_ROWS || C_COLS != B_COLS) {
//         printf("output matrix dimension mismatch\r\n");
//         return;
//     }

//     for(int r = 0; r < A_ROWS; r++) {
//         for(int c = 0; c < B_COLS; c++) {

//             uint32_t acc = 0;

//             for(int i = 0; i < B_ROWS; i++) {
//                 acc += A_Data[i + r * A_COLS] * B_Data[c + i * B_ROWS];
//             }

//             C_Data[c + r * C_ROWS] = acc;
//         }
//     }
// }

// tensorflow/tensorflow/core/kernels/reference_gemm.h

// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by TensorFlow, and implements
// C = A * B, like the standard BLAS GEMM interface. If the transpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

//     template <class T1, class T2, class T3>
//     void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
//                        size_t m, size_t n, size_t k, const T1* a, int32
//                        offset_a, size_t lda, const T2* b, int32 offset_b,
//                        size_t ldb, T3* c, int32 shift_c, int32 offset_c,
//                        int32 mult_c, size_t ldc) {
//       int a_i_stride;
//       int a_l_stride;
//       if (transpose_a) {
//         a_i_stride = 1;
//         a_l_stride = lda;
//       } else {
//         a_i_stride = lda;
//         a_l_stride = 1;
//       }
//       int b_j_stride;
//       int b_l_stride;
//       if (transpose_b) {
//         b_j_stride = ldb;
//         b_l_stride = 1;
//       } else {
//         b_j_stride = 1;
//         b_l_stride = ldb;
//       }
//       int c_i_stride;
//       int c_j_stride;
//       if (transpose_c) {
//         c_i_stride = 1;
//         c_j_stride = ldc;
//       } else {
//         c_i_stride = ldc;
//         c_j_stride = 1;
//       }

//     //   const int32 highest =
//     static_cast<int32>(Eigen::NumTraits<T3>::highest());
//     //   const int32 lowest =
//     static_cast<int32>(Eigen::NumTraits<T3>::lowest());
//       const int32 highest =
//       static_cast<int32>(std::numeric_limits<T3>::max()); const int32 lowest
//       = static_cast<int32>(std::numeric_limits<T3>::min()); const int32
//       rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

//       int i, j, l;
//       for (j = 0; j < n; j++) {
//         for (i = 0; i < m; i++) {
//           int32 total = 0;
//           for (l = 0; l < k; l++) {
//             const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
//             const int32 a_value = static_cast<int32>(a[a_index]) - offset_a;
//             const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
//             const int32 b_value = static_cast<int32>(b[b_index]) - offset_b;
//             total += (a_value * b_value);
//           }
//           const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
//           int32_t output = ((((total + offset_c) * mult_c) + rounding) >>
//           shift_c); if (output > highest) {
//             output = highest;
//           }
//           if (output < lowest) {
//             output = lowest;
//           }
//           c[c_index] = static_cast<T3>(output);
//         }
//       }
//     }

template <class T1, class T2, class T3>
void ReferenceGemmuImpl(bool transpose_a, bool transpose_b, bool transpose_c,
                        size_t m, size_t n, size_t k, const T1* a,
                        int32_t offset_a, size_t lda, const T2* b, int offset_b,
                        size_t ldb, T3* c, int shift_c, int offset_c,
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
        const int32_t a_value = static_cast<int32_t>(a[a_index]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const int32_t b_value = static_cast<int32_t>(b[b_index]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }

      if (output < lowest) {
        output = lowest;
      }
      c[c_index] = static_cast<T3>(output);
    }
  }
}

template <class T>
int64_t FloatToQuantizedUnclamped(float input, float range_min,
                                  float range_max) {
  const int64_t lowest_quantized =
      static_cast<double>(std::numeric_limits<T>::lowest());
  if (range_min == range_max) {
    return lowest_quantized;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale =
      (number_of_steps /
       range);  // NT: fractional resolution?  bit per floating-point-increment
  int64_t quantized =
      (round(input * range_scale) -
       round(range_min *
             range_scale));  // NT: roughly: (input - range_min) * range_scale
  quantized += lowest_quantized;  // NT: can be a negative number for <signed
                                  // ...>; zero gets down-shifted to
                                  // Eigen::NumTraits<T>::lowest()
  return quantized;  // NT: input<float> qanuntized to a specific range<int64>
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
void QuantizedMatMul(Tensor<T1> A, Tensor<T2> B, Tensor<Toutput> C,
                     Tensor<float> mina, Tensor<float> minb, Tensor<float> maxa,
                     Tensor<float> maxb, Tensor<float> outmin,
                     Tensor<float> outmax) {
  const float min_a = *(mina.getPointer({}));
  const float max_a = *(maxa.getPointer({}));
  const float min_b = *(minb.getPointer({}));
  const float max_b = *(maxb.getPointer({}));
  bool transpose_a = false;
  bool transpose_b = false;

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

  T1* A_Data = A.getPointer({});
  T2* B_Data = B.getPointer({});
  Toutput* C_Data = C.getPointer({});

  const bool transpose_c = false;
  const size_t m = A.getShape()[a_dim_remaining];
  const size_t n = B.getShape()[b_dim_remaining];
  const size_t k = A.getShape()[first];
  const size_t lda = A.getShape()[1];
  const size_t ldb = B.getShape()[1];
  const size_t ldc = n;

  ReferenceGemmuImpl<T1, T2, Toutput>(
      transpose_a, transpose_b, transpose_c, m, n, k, A_Data, offset_a, lda,
      B_Data, offset_b, ldb, C_Data, shift_c, offset_c, mult_c, ldc);
  float min_c_value;
  float max_c_value;

  QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);

  float* c_min = outmin.getPointer({});
  *c_min = min_c_value;
  float* c_max = outmax.getPointer({});
  *c_max = max_c_value;
}

class matrixOpsTest : public Test {
 public:
  void qMatMul(void) {
    testStart("Quantized Matrix Mul");
    TensorIdxImporter t_import;

    // reference inputs
    Tensor<unsigned char> a =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx");
    Tensor<float> a_min =
        t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx");
    Tensor<float> a_max =
        t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx");
    Tensor<unsigned char> b =
        t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx");
    Tensor<float> b_min =
        t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx");
    Tensor<float> b_max =
        t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");

    // reference outputs
    Tensor<int> c =
        t_import.int_import("/fs/testData/qMatMul/out/qMatMul_0.idx");
    Tensor<float> c_min =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_1.idx");
    Tensor<float> c_max =
        t_import.float_import("/fs/testData/qMatMul/out/qMatMul_2.idx");

    // actual implementation, uses ReferenceGemm()
    // See gen_math_op.py:1619
    // See quantized_matmul_ops.cc:171, 178
    // Sub-functions: QuantizationRangeForMultiplication,
    // QuantizationRangeForMultiplication, FloatForOneQuantizedLevel

    Tensor<int> out_c(c.getShape());
    Tensor<float> out_min(c_min.getShape());
    Tensor<float> out_max(c_max.getShape());
    QuantizedMatMul<uint8_t, uint8_t, int>(a, b, out_c, a_min, b_min, a_max,
                                           b_max, out_min, out_max);
    //
    // transpose_a=None, transpose_b=None

    // modify the checks below:

    double result = meanPercentErr(c, out_c) + meanPercentErr(c_min, out_min) +
                    meanPercentErr(c_max, out_max);
    // passed(result < 0.0001);
    passed(result == 0);
  }

  void runAll(void) { qMatMul(); }
};

#endif
