#ifndef UTENSOR_BASE_H
#define UTENSOR_BASE_H

#include "tensor.hpp"

typedef long long TensorPtr;
typedef vector<Tensor*> TList;

class uTensor {
  virtual void inFocus() {};
  virtual void deFocus() {};
  virtual ~uTensor() = 0;
};


//isType() https://stackoverflow.com/questions/9974596/how-to-check-whether-two-pointers-point-to-the-same-object-or-not
//double dispatch

//new vs stack
class Operator {
protected:
  //setup input/output info in derived constructors
  TList inputs;
  vector<DType> dtype_in;
  TList outputs;
  vector<DType> dtype_out;
public:
  virtual void compute() = 0;

  void setInputs(TList &_inputs) {
    if(_inputs.size() != inputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < input.size(); i++) {
      if(dtype_in[i] != inputs.getType()) {
        ERR_EXIT("Tensor Type mismatched...");
      }

      input[i] = _inputs[i];
    }
  }

  void setOutputs(TList &_outputs) {
    if(_outputs.size() != outputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < output.size(); i++) {
      if(dtype_out[i].getType() != output[i].getType()) {
        ERR_EXIT("Tensor Type mismatched...");
      }

      output[i] = _output[i]
    }
  }

  TList getInputs(void) {
    return inputs;
  }

  TList getOutputs(void) {
    return outputs;
  }
};


#endif //UTENSOR_BASE_H
