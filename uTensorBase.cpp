#include "uTensorBase.hpp"

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
