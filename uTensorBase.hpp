#ifndef UTENSOR_BASE_H
#define UTENSOR_BASE_H

#include "tensor.hpp"

typedef vector<Tensor*> TList;

//isType() https://stackoverflow.com/questions/9974596/how-to-check-whether-two-pointers-point-to-the-same-object-or-not
//double dispatch

//new vs stack
class Operator : public uTensor{
protected:
  //setup input/output info in derived constructors
  //ref count?
  TList inputs;
  vector<DType> dtype_in;
  TList outputs;
  vector<DType> dtype_out;
public:
  virtual void compute() = 0;

  void setInputs(TList &_inputs) {
    if(_inputs.size() != inputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < inputs.size(); i++) {
      if(dtype_in[i] != inputs[i]->getDType()) {
        ERR_EXIT("Tensor Type mismatched...");
      }

      inputs[i] = _inputs[i];
    }
  }

  void setOutputs(TList &_outputs) {
    if(_outputs.size() != outputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < outputs.size(); i++) {
      if(dtype_out[i] != outputs[i]->getDType()) {
        ERR_EXIT("Tensor Type mismatched...");
      }

      outputs[i] = _outputs[i];
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
