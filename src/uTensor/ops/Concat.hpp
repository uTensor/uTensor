#include "uTensor/ops/Concat_kernels.hpp"

namespace uTensor {
namespace ReferenceOperators {

class ConcatOperator : public OperatorInterface<3, 1> {
 public:
  // identifiers for setting up the input tensors
  enum names_in { a, b, axis };
  // identifiers for setting up the output tensors
  enum names_out { out };

 protected:
  void compute() {
    // you can retrieve input/output tensors by its identifier
    concat_kernel(inputs[a].tensor(), inputs[b].tensor(), inputs[axis].tensor(),
                  outputs[out].tensor());
  }
};

}  // namespace ReferenceOperators
}  // namespace uTensor
