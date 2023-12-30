#include <pybind11/numpy.h>

#include "uTensor.h"
#include "uTensor/core/operatorBase.hpp"
#include "uTensor/ops/Matrix.hpp"

using uTensor::FastOperator;
using uTensor::Tensor;
namespace py = pybind11;
class CopyOperator : public FastOperator {
 public:
  void toTensor(const void *src, Tensor &dest);
  void fromTensor(void *dest, const Tensor &src);
  py::buffer_info getInfo(const Tensor &tensor);
};
