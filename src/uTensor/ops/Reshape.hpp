#ifndef UTENSOR_RESHAPE_H
#define UTENSOR_RESHAPE_H

#include "context.hpp"
#include "types.hpp"
#include "tensor.hpp"
#include "uTensor_util.hpp"
#include "operatorBase.hpp"

namespace uTensor {
template <typename Tin>
class ReshapeOperator : public OperatorInterface<1, 1> {
/* reshape input as the shape of output*/
public:
  enum names_in : unit8_t { input };
  enum names_out : uint8_t { output };
  virtual void compute(){
    Tensor input_tensor = inputs[input];
    Tensor output_tensor = outputs[output];
    if (input_tensor->num_elems() != output_tensor->num_elems()){
      uTensor_printf("inconsistent input and output shape for reshape\n");
      Context::get_default_context()->throwError(new InvalidReshapeError);
      return;
    }
    if (input_tensor->get_type() != output_tensor->get_type()){
      uTensor_printf("inconsistent input and output data type for reshape\n");
      Context::get_default_context()->throwError(new InvalidTensorDataTypeError);
      return;
    }
    input_tensor->get_shape() = output_tensor->get_shape();
  }
};
}

#endif // UTENSOR_RESHAPE_H