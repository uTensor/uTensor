#ifndef UTENSOR_BASE_H
#define UTENSOR_BASE_H

#include "src/uTensor/core/tensor.hpp"

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
  void setInputs(S_TList &_inputs);
  void setOutputs(S_TList &_outputs);
  S_TList getInputs(void) { return inputs; }
  S_TList getOutputs(void) { return outputs;}
  uint8_t getNumInputs(void) { return n_inputs; }
  uint8_t getNumOutputs(void) { return n_outputs; }
  void empty(void);

  Operator() {
    n_inputs = 0;  //overridden by constructor
    n_outputs = 0;
  }
};


#endif //UTENSOR_BASE_H
