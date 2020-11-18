#ifndef UTENSOR_TRANSPOSE_H
#define UTENSOR_TRANSPOSE_H

#include "context.hpp"
#include "types.hpp"
#include "tensor.hpp"
#include "uTensor_util.hpp"
#include "operatorBase.hpp"

#include <cstring>

namespace uTensor {
namespace ReferenceOperators {

// Transpose (Swap Axes) as a port from Numpy
// using stride interation in the order of transpose axes
template <typename Tin>
class TransposeOperator : public OperatorInterface<1, 1> {
/* reshape input as the shape of output*/
public:
  TransposeOperator(const TensorShape&& axes) : _axes(axes) {}
  TransposeOperator(const TensorShape& axes) : _axes(axes) {}
  
  enum names_in : uint8_t { input };
  enum names_out : uint8_t { output };

  virtual void compute(){
    Tensor& input_tensor = inputs[input].tensor();
    TensorShape& input_shape = input_tensor.get_shape();
    input_shape.update_dims();

    // Strides are used to iterate over the dataset, and transfer
    // the input tensor data, into the output tensor
    TensorStrides input_strides = TensorStrides(input_shape);

    Tensor& output_tensor = outputs[output].tensor();

    // Create a placeholder to calculate the output shape
    // Normally this would reference output shape, but since this could (usually would) be referencing the input, let's keep a dedicated value
    TensorShape output_shape = TensorShape(1,1,1,1);
    TensorStrides output_strides = TensorStrides(output_shape);
    TensorShape offsets = TensorShape(input_shape.num_dims());

    for (size_t i = 0; i < 4; ++i) { 
      output_shape[i] = 0;
      output_strides[i] = 0;

      // Offsets are used to avoid multiple for loops
      offsets[i] = 0;
    }

    for (size_t i = 0; i < (size_t) input_shape.num_dims(); ++i) { 
      output_shape[_axes[i]] = input_shape[i];

      // output_strides(i) is derived from axes and input_strides
      output_strides[_axes[i]] = (*input_strides)[i];
    }
    
    // Output shape can be asserted once the transform 
    // effect has been determined
    output_shape->update_dims();
    output_tensor->resize(output_shape);

    // Perform some basic checks
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
    if (!_check_input_shape()){
        Context::get_default_context()->throwError(new InvalidTensorDataTypeError);
        return;
    }

    // copy data
    for (uint32_t i = 0; i < input_tensor->num_elems(); ++i) { 
        // Index of the source value, must be calculated
        // using the output strides and output shape
        uint32_t idx = 0;
        for (uint32_t j = 0; j < output_shape->num_dims(); j++) {
            idx += offsets[j] * output_strides[j];
        }

        // this is not copy: `output_tensor(i) = input_tensor(i);`
        output_tensor(i) = static_cast<Tin>(input_tensor(idx));

        // Update offsets, to iterate sequentially along strides
        // in the order of axes
        for (int32_t j = output_shape->num_dims() - 1; j >= 0; j--) {
            offsets[j] = (offsets[j] + 1) % (output_shape[j]);
            if( offsets[j] > 0 ) {
                break;
            }
        }        
    }  

  }
private:
  TensorShape _axes;

  bool _check_input_shape(){
    const Tensor& input_tensor = inputs[input].tensor();
    const TensorShape& shape = input_tensor->get_shape();
    uint8_t num_dims = shape.num_dims();
    for (int i = 0; i < num_dims; ++i){
      if (shape[i] < 0) {
        uTensor_printf("the output shape must be all positive\n");
        return false;
      }
    }
    return true;
  }
};

}
}

#endif // UTENSOR_TRANSPOSE_H
