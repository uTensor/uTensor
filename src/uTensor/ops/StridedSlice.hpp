#ifndef UTENSOR_STRIDED_SLICE_OPS_H
#define UTENSOR_STRIDED_SLICE_OPS_H
#include <algorithm>
#include <limits>

#include "StridedSlice_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename T>
class StridedSliceOperator : public OperatorInterface<4, 1> {
 private:
  int begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask;

 public:
  enum names_in : uint8_t { input, begin, end, strides };
  enum names_out : uint8_t { output };

  StridedSliceOperator(int begin_mask_, int end_mask_, int ellipsis_mask_,
                       int new_axis_mask_, int shrink_axis_mask_)
      : begin_mask(begin_mask_),
        end_mask(end_mask_),
        ellipsis_mask(ellipsis_mask_),
        new_axis_mask(new_axis_mask_),
        shrink_axis_mask(shrink_axis_mask_) {}

 protected:
  virtual void compute() {
    stridedslice_kernel<T>(inputs[input].tensor(), inputs[begin].tensor(),
                           inputs[end].tensor(), inputs[strides].tensor(),
                           outputs[output].tensor(), begin_mask, end_mask,
                           ellipsis_mask, new_axis_mask, shrink_axis_mask);
  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor
#endif  // UTENSOR_STRIDED_SLICE_OPS_H