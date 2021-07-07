#include "uTensor/ops/Concat_kernels.hpp"

namespace uTensor {
namespace ReferenceOperators {

template <typename T>
class ConcatOperator : public OperatorInterface<3, 1> {
 public:
  ConcatOperator(int32_t axis) : _axis(axis) {}
  // identifiers for setting up the input tensors
  enum names_in { a, b };
  // identifiers for setting up the output tensors
  enum names_out { out };

 protected:
  void compute() {
    // you can retrieve input/output tensors by its identifier
    concat_kernel<T>(inputs[a].tensor(), inputs[b].tensor(), _axis,
                     outputs[out].tensor());
  }

 private:
  int32_t _axis;
};

}  // namespace ReferenceOperators
}  // namespace uTensor
