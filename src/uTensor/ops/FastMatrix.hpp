#ifndef __UTENSOR_FASTMATRIX_HPP
#define __UTENSOR_FASTMATRIX_HPP
#include "uTensor/core/operatorBase.hpp"

namespace uTensor {

template <typename T>
class FastMatMulOperator : public OperatorInterface<2, 1>, FastOperator {
 public:
  enum names : uint8_t { input1, input2, output };

  virtual void compute() {
    Tensor a = inputs[input1].tensor();
    Tensor b = inputs[input2].tensor();
    Tensor c = outputs[output].tensor();
    TensorShape a_shape = a->get_shape();  // M x K
    TensorShape b_shape = b->get_shape();  // K x N
    TensorShape c_shape = c->get_shape();  // M x N

    T *a_ptr, *b_ptr, *c_ptr;
    get_readable_block(a, a_ptr, a_shape.num_elems(), 0);
    get_readable_block(b, b_ptr, b_shape.num_elems(), 0);
    get_writeable_block(c, c_ptr, c_shape.num_elems(), 0);
    fast_matmul_kernel<T>(a_shape, b_shape, c_shape a_ptr, b_ptr, c_ptr);
  }
};

}  // namespace uTensor

#endif  //__UTENSOR_FASTMATRIX_HPP