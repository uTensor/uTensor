#ifndef __UTENSOR_FASTMATRIX_HPP
#define __UTENSOR_FASTMATRIX_HPP
#include "uTensor/core/operatorBase.hpp"

namespace uTensor {

template <typename T>
class FastMatMulOperator : public OperatorInterface<2, 1>, FastOperator {
 public:
  enum names : uint8_t { filter, input, output };

  virtual void compute() {
    Tensor w = inputs[filter].tensor();
    Tensor in = inputs[input].tensor();
    Tensor out = outputs[output].tensor();
    TensorShape w_shape = w->get_shape();      // batch x M x K
    TensorShape in_shape = in->get_shape();    // batch x K x N
    TensorShape out_shape = out->get_shape();  // batch x M x N

    T *w_ptr, *in_ptr, *out_ptr;
    get_readable_block(w, w_ptr, w_shape.num_elems(), 0);
    get_readable_block(in, in_ptr, in_shape.num_elems(), 0);
    get_writeable_block(out, out_ptr, out_shape.num_elems(), 0);
    fast_matmul_kernel<T>(w_shape, in_shape, out_shape w_ptr, in_ptr, out_ptr);
  }
};

}  // namespace uTensor

#endif  //__UTENSOR_FASTMATRIX_HPP