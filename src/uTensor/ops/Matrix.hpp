#ifndef UTENSOR_MATRIX_OPS_H
#define UTENSOR_MATRIX_OPS_H
#include "context.hpp"
#include "operatorBase.hpp"
#include "ActivationFncs.hpp"
#include "Matrix_kernels.hpp"

namespace uTensor {
namespace ReferenceOperators {

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

template <typename T>
class MatrixMultOperatorV2 : public OperatorInterface<3, 1> {};

template <>
class MatrixMultOperatorV2<float> : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { input, filter, bias };
  enum names_out : uint8_t { output };

  MatrixMultOperatorV2(Fuseable::Activation<float> activation = Fuseable::NoActivation<float>)
      : _activation(activation) {}

 private:
  Fuseable::Activation<float> _activation;

 protected:
  virtual void compute() {
    bool have_bias =
        *(inputs[bias].name) != *(TensorMapInterface::not_found.name);
    // Decide on c shape
    TensorShape& a_shape = inputs[input].tensor()->get_shape();
    TensorShape& b_shape = inputs[filter].tensor()->get_shape();
    TensorShape& c_shape = outputs[output].tensor()->get_shape();
    if (a_shape.num_dims() > 2 || b_shape.num_dims() > 2 ||
        c_shape.num_dims() > 2 || a_shape[1] != b_shape[0] ||
        a_shape[0] != c_shape[0] || b_shape[1] != c_shape[1]) {
      uTensor_printf("[Error] Invalid matrix multiple shape mismatch\n");
      Context::get_default_context()->throwError(
          new InvalidMatrixMultIndicesError);
    }
    if (have_bias) {
      matrix_mult_kernel_v2<float>(
          outputs[output].tensor(), inputs[input].tensor(),
          inputs[filter].tensor(), inputs[bias].tensor(), _activation);
    } else {
      matrix_mult_kernel_v2<float>(
          outputs[output].tensor(), inputs[input].tensor(),
          inputs[filter].tensor(), _activation);
    }
  }
};

template <typename Tout>
using FullyConnectedOperator = MatrixMultOperatorV2<Tout>;

}
}  // namespace uTensor
#endif
