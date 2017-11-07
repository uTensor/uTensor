#ifndef UTENSOR_BASE_H
#define UTENSOR_BASE_H

#include "tensor.hpp"

typedef std::vector<Tensor*> TList;

//isType() https://stackoverflow.com/questions/9974596/how-to-check-whether-two-pointers-point-to-the-same-object-or-not
//double dispatch

//new vs stack
class Operator : public uTensor{
protected:
  //setup input/output info in derived constructors
  //ref count?
  TList inputs;
  uint8_t n_inputs;
  TList outputs;
  uint8_t n_outputs;

public:
  virtual void compute() = 0;
  void setInputs(TList &_inputs);
  void setOutputs(TList &_outputs);
  TList getInputs(void) { return inputs; }
  TList getOutputs(void) { return outputs;}
  uint8_t getNumInputs(void) { return n_inputs; }
  uint8_t getNumOutputs(void) { return n_outputs; }

  Operator() {
    n_inputs = 0;  //overridden by constructor
    n_outputs = 0;
  }
};


#endif //UTENSOR_BASE_H
