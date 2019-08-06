#include "src/uTensor/core/uTensorBase.hpp"

void Operator::setInputs(S_TList &_inputs) {
  if(_inputs.size() != n_inputs) ERR_EXIT("Input Tensor list mismatched...");

  inputs = _inputs;
}

void Operator::setOutputs(S_TList &_outputs) {
  if(_outputs.size() != n_outputs) ERR_EXIT("Output Tensor list mismatched...");

  outputs = _outputs;

}

void Operator::empty(void) {
  inputs.empty();
  outputs.empty();
}
