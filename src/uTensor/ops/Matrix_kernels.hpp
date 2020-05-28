#ifndef UTENSOR_MATRIX_KERNELS
#define UTENSOR_MATRIX_KERNELS
#include "context.hpp"
#include "operatorBase.hpp"
#include "ActivationFncs.hpp"

namespace uTensor {
DECLARE_ERROR(InvalidMatrixMultIndicesError);

// Assume c is already allocated to the correct size
// Naive implementation
template <typename T>
void matrix_mult_kernel(Tensor& c, const Tensor& a, const Tensor& b) {
  // Decide on c shape
  TensorShape a_shape = a->get_shape();
  TensorShape b_shape = b->get_shape();
  TensorShape c_shape = c->get_shape();
  if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
      c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
      a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
    uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }

  for (uint32_t i = 0; i < a_shape[0]; i++) {
    for (uint32_t j = 0; j < b_shape[1]; j++) {
      // c(i, j) = static_cast<T>(0);
      T tmp = 0;
      for (uint32_t k = 0; k < a_shape[1]; k++) {
        tmp += static_cast<T>(a(i, k)) * static_cast<T>(b(k, j));
        // printf("i, j, k : %d %d %d %d %d\n", i, j, k, static_cast<T>(a(i, k))
        // , static_cast<T>(b(k, j)));
      }
      c(i, j) = tmp;
    }
  }
}

template <typename T>
void matrix_mult_kernel_v2(Tensor& output, const Tensor& input,
                                  const Tensor& filter,
                                  Fuseable::Activation<T> activation){
  const TensorShape& input_shape = input->get_shape();
  const TensorShape& filter_shape = filter->get_shape();
  TensorShape& output_shape = output->get_shape();

  const int filter_dim_count = filter_shape.num_dims();
  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  if (!(output_depth < filter_shape[filter_dim_count - 2])) {
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  const int accum_depth = filter_shape[0];
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      T acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        // TODO write this in tensor form
        T input_val = static_cast<T>(input(b, d, 0, 0));
        T filter_val = static_cast<T>(filter(d, out_c, 0, 0));
        acc += filter_val * input_val;
      }
      acc = activation(acc);
      output(b, out_c, 0, 0) = static_cast<T>(acc);
    }
  }
}

template <typename T>
void matrix_mult_kernel_v2(Tensor& output, const Tensor& input,
                                  const Tensor& filter, const Tensor& bias,
                                  Fuseable::Activation<T> activation){
  const TensorShape& input_shape = input->get_shape();
  const TensorShape& filter_shape = filter->get_shape();
  TensorShape& output_shape = output->get_shape();

  const int filter_dim_count = filter_shape.num_dims();
  const int batches = output_shape[0];
  const int output_depth = output_shape[1];
  if (!(output_depth < filter_shape[filter_dim_count - 2])) {
    Context::get_default_context()->throwError(
        new InvalidMatrixMultIndicesError);
  }
  const int accum_depth = filter_shape[0];
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      T acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        // TODO write this in tensor form
        T input_val = static_cast<T>(input(b, d, 0, 0));
        T filter_val = static_cast<T>(filter(d, out_c, 0, 0));
        acc += filter_val * input_val;
      }
      acc += static_cast<T>(bias(out_c));
      acc = activation(acc);
      output(b, out_c, 0, 0) = static_cast<T>(acc);
    }
  }
}

}
#endif
