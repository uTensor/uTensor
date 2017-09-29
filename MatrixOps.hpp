#ifndef UTENSOR_MATRIX_OPS
#define UTENSOR_MATRIX_OPS

#include <limits>

//Useful links
//http://www.plantation-productions.com/Webster/www.artofasm.com/Linux/HTML/Arraysa2.html

//Assume Tensor index order:  rows, columns | the first 2 dimensions
typedef Tensor<unsigned char> Mat

void printMat(Mat mat) {
    const uint32_t ROWS = mat.getShape()[0];
    const uint32_t COLS = mat.getShape()[1];
    unsigned char* pData = mat.getPointer({});

    for(int i_r = 0; i_r < ROWS; i_r++) {
        for(int i_c = 0; i_c < COLS; i_c++) {
            printf("%f ", pData[i_r * COLS + i_c]);
        }
        printf("\r\n");
    }
}

void initMatConst(Mat mat, unsigned char value) {
    const uint32_t ROWS = mat.getShape()[0];
    const uint32_t COLS = mat.getShape()[1];
    unsigned char* pData = mat.getPointer({});

    for(int i_r = 0; i_r < ROWS; i_r++) {
        for(int i_c = 0; i_c < COLS; i_c++) {
            pData[i_r * COLS + i_c] = value;
        }
    }
}

void multMat(Mat A, Mat B, Mat C) {
    const uint32_t A_ROWS = A.getShape()[0];
    const uint32_t A_COLS = A.getShape()[1];
    const uint32_t B_ROWS = B.getShape()[0];
    const uint32_t B_COLS = B.getShape()[1];
    const uint32_t C_ROWS = C.getShape()[0];
    const uint32_t C_COLS = C.getShape()[1];
    unsigned char* A_Data = A.getPointer({});
    unsigned char* B_Data = B.getPointer({});
    unsigned char* C_Data = C.getPointer({});

    if(A_COLS != B_ROWS) {
        printf("A and B matrices dimension mismatch\r\n");
        return;
    }

    if(C_ROWS != A_ROWS || C_COLS != B_COLS) {
        printf("output matrix dimension mismatch\r\n");
        return;
    }

    for(int r = 0; r < A_ROWS; r++) {
        for(int c = 0; c < B_COLS; c++) {

            uint32_t acc = 0;

            for(int i = 0; i < B_ROWS; i++) {
                acc += A_Data[i + r * A_COLS] * B_Data[c + i * B_ROWS];
            }

            C_Data[c + r * C_ROWS] = acc;
        }
    }
}

//tensorflow/tensorflow/core/kernels/reference_gemm.h

// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by TensorFlow, and implements
// C = A * B, like the standard BLAS GEMM interface. If the transpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

    template <class T1, class T2, class T3>
    void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                       size_t m, size_t n, size_t k, const T1* a, int32 offset_a,
                       size_t lda, const T2* b, int32 offset_b, size_t ldb, T3* c,
                       int32 shift_c, int32 offset_c, int32 mult_c, size_t ldc) {
      int a_i_stride;
      int a_l_stride;
      if (transpose_a) {
        a_i_stride = 1;
        a_l_stride = lda;
      } else {
        a_i_stride = lda;
        a_l_stride = 1;
      }
      int b_j_stride;
      int b_l_stride;
      if (transpose_b) {
        b_j_stride = ldb;
        b_l_stride = 1;
      } else {
        b_j_stride = 1;
        b_l_stride = ldb;
      }
      int c_i_stride;
      int c_j_stride;
      if (transpose_c) {
        c_i_stride = 1;
        c_j_stride = ldc;
      } else {
        c_i_stride = ldc;
        c_j_stride = 1;
      }
    
    //   const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
    //   const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());
      const int32 highest = static_cast<int32>(std::numeric_limits<T3>::max());
      const int32 lowest = static_cast<int32>(std::numeric_limits<T3>::min());
      const int32 rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));
    
      int i, j, l;
      for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
          int32 total = 0;
          for (l = 0; l < k; l++) {
            const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
            const int32 a_value = static_cast<int32>(a[a_index]) - offset_a;
            const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
            const int32 b_value = static_cast<int32>(b[b_index]) - offset_b;
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

template <class T1, class T2, class T3>
void ReferenceGemmuImpl(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, const T1* a, int32 offset_a,
                   size_t lda, const T2* b, int32 offset_b, size_t ldb, T3* c,
                   int32 shift_c, int32 offset_c, int32 mult_c, size_t ldc) {

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

    //   const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
    //   const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());
    const int32 highest = static_cast<int32>(std::numeric_limits<T3>::max());
    const int32 lowest = static_cast<int32>(std::numeric_limits<T3>::min());
    const int32 rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

    int i, j, l;
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        int32 total = 0;
        for (l = 0; l < k; l++) {
          const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
          const int32 a_value = static_cast<int32>(a[a_index]) - offset_a;
          const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
          const int32 b_value = static_cast<int32>(b[b_index]) - offset_b;
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
    

#endif
