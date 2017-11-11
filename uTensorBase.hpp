#ifndef UTENSOR_BASE_H
#define UTENSOR_BASE_H

#include "tensor.hpp"

//isType() https://stackoverflow.com/questions/9974596/how-to-check-whether-two-pointers-point-to-the-same-object-or-not
//double dispatch

//new vs stack
class Operator : public uTensor {
protected:
  //setup input/output info in derived constructors
  //ref count?
  S_TList inputs;
  uint8_t n_inputs;
  S_TList outputs;
  uint8_t n_outputs;

public:
  virtual void compute() = 0;
  void setInputs(TList &_inputs);
  void setOutputs(TList &_outputs);
  S_TList getInputs(void) { return inputs; }
  S_TList getOutputs(void) { return outputs;}
  uint8_t getNumInputs(void) { return n_inputs; }
  uint8_t getNumOutputs(void) { return n_outputs; }

  Operator() {
    n_inputs = 0;  //overridden by constructor
    n_outputs = 0;
  }
};

void Operator::setInputs(TList &_inputs) {
  if(_inputs.size() != n_inputs) ERR_EXIT("Input Tensor list mismatched...");

  for(uint8_t i=0; i < _inputs.size(); i++) {
    inputs.push_back(_inputs[i].lock());
  }
}

void Operator::setOutputs(TList &_outputs) {
  if(_outputs.size() != n_outputs) ERR_EXIT("Input Tensor list mismatched...");

  for(uint8_t i=0; i < _outputs.size(); i++) {
    outputs.push_back(_outputs[i].lock());
  }
}

#endif //UTENSOR_BASE_H
