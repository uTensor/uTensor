#ifndef UTENSOR_ARITHMETIC_OPS_H
#define UTENSOR_ARITHMETIC_OPS_H
#include <algorithm>
#include <limits>

#include "StridedSlice_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename T>
class StridedSliceOperator : public OperatorInterface<9, 1> {
 public:
  enum names_in : uint8_t { i, b, e, s, b_m, ell_m, e_m, n_a_m, s_a_m };
  enum names_out : uint8_t { o };

 protected:
  virtual void compute() {
    stridedslice_kernel<T>(inputs[i].tensor(), inputs[b].tensor(),
                           inputs[e].tensor(), inputs[s].tensor(),
                           outputs[o].tensor(), inputs[b_m].tensor(),
                           inputs[ell_m].tensor(), inputs[e_m].tensor(),
                           inputs[n_a_m].tensor(), inputs[s_a_m].tensor());
  }
};
}  // namespace ReferenceOperators
}  // namespace uTensor
#endif