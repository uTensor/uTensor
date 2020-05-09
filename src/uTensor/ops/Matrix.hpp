#ifndef UTENSOR_MATRIX_OPS_H
#define UTENSOR_MATRIX_OPS_H
#include "context.hpp"
#include "operatorBase.hpp"

namespace uTensor {

DECLARE_ERROR(InvalidMatrixMultIndicesError);
namespace ReferenceOperators {

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
class MatrixMultOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { a, b };
  enum names_out : uint8_t { c };
  // AddOperator(FixedTensorMap<2> inputs, FixedTensorMap<1> outputs) :
  // OperatorBase(inputs, outputs) {}

 protected:
  virtual void compute() {
    matrix_mult_kernel<T>(outputs[c].tensor(), inputs[a].tensor(),
                          inputs[b].tensor());
  }
};

}
}  // namespace uTensor
#endif
