#ifndef __UTENSOR_FASTMATRIX_KERNELS_HPP
#define __UTENSOR_FASTMATRIX_KERNELS_HPP
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/ops/Matrix_kernels.hpp"

namespace uTensor {

template <typename T>
void fast_matmul_kernel(TensorShape w_shape, TensorShape in_shape,
                        TensorShape out_shape, const T *ptr_w, const T *ptr_in,
                        T *ptr_out) {
  // check shapes
  if (w_shape.num_dims() != 3 || in_shape.num_dims() != 3 ||
      out_shape.num_dims() != 3) {
    uTensor_printf("[Error] incorrect number of dimensions\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  if (w_shape[1] != out_shape[1] || w_shape[2] != in_shape[1] ||
      in_shape[2] != out_shape[2]) {
    uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  uint16_t batch_size = out_shape[0], M = w_shape[0], K = w_shape[1],
           N = in_shape[2];
  for (uint16_t b = 0; b < batch_size; b++) {
    for (uint16_t i = 0; i < M; i++) {
      for (uint16_t k = 0; k < K; k++) {
        auto idx_a = w_shape.linear_index(b, i, k);
        uint32_t a_ik = *ptr_w[idx_a];
        for (uint16_t j = 0; j < N; j++) {
          uint32_t idx_b = in_shape.linear_index(b, k, j),
                   idx_c = out_shape.linear_index(b, i, j);
          T b_kj = *ptr_in[idx_b];
          T c_ij = *ptr_out[idx_c];
          c_ij += a_ik * b_kj;
          ptr_out[idx_c] = c_ij;
        }
      }
    }
  }
}
}  // namespace uTensor

#endif