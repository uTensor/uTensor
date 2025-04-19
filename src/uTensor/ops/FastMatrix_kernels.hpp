#ifndef __UTENSOR_FASTMATRIX_KERNELS_HPP
#define __UTENSOR_FASTMATRIX_KERNELS_HPP
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/ops/Matrix_kernels.hpp"

namespace uTensor {

template <typename T>
void fast_matmul_kernel(TensorShape a_shape, TensorShape b_shape,
                        TensorShape c_shape, const T *ptr_a, const T *ptr_b,
                        T *ptr_c) {
  // check shapes
  if (a_shape.num_dims() != 2 || b_shape.num_dims() != 2 ||
      c_shape.num_dims() != 2 || a_shape[1] != b_shape[0] ||
      a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
    uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  uint16_t M = a_shape[0], K = a_shape[1], K = b_shape[0];
  for (uint16_t i = 0; i < M; i++) {
    for (uint16_t k = 0; k < K; k++) {
      auto idx_a = a_shape.linear_index(i, k);
      uint32_t a_ik = *ptr_a[idx_a];
      for (uint16_t j = 0; j < N; j++) {
        uint32_t idx_b = b_shape.linear_index(k, j),
                 idx_c = c_shape.linear_index(i, j);
        T b_kj = *ptr_b[idx_b];
        T c_ij = *ptr_c[idx_c];
        c_ij += a_ik * b_kj;
        ptr_c[idx_c] = c_ij;
      }
    }
  }
}
}  // namespace uTensor

#endif