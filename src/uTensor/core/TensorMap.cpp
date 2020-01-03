#include "TensorMap.hpp"
namespace uTensor {
  TensorMapInterface::SimpleNamedTensor not_found(uTensor::string("NotFound"),
                                     static_cast<Tensor>(NULL));
}
