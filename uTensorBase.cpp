#include "uTensorBase.hpp"

void Operator::setInputs(TList &_inputs) {
  if(_inputs.size() != n_inputs) ERR_EXIT("Input Tensor list mismatched...");

  inputs = _inputs;
}

void Operator::setOutputs(TList &_outputs) {
  if(_outputs.size() != n_outputs) ERR_EXIT("Input Tensor list mismatched...");

  outputs = _outputs;
}
