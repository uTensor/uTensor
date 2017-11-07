#include "context.hpp"

void Context::push(Operator *op, TList &_inputs, TList &_outputs) {
  if(op->getNumInputs() != _inputs.size()) {
    ERR_EXIT("valid number of inputs\r\n");
  }
  if(op->getNumOutputs() != _outputs.size()) {
    ERR_EXIT("valid number of output\r\n");
  }

  op->setInputs(_inputs);
  op->setOutputs(_outputs);
  op_list.push_back(op);
  updateInputTensorRef(_inputs);

}

void Context::updateInputTensorRef(const TList &t_list) {
  for(auto t:t_list) {
    t->incrRef();  //if an initial ref value is supplied to the tensor at compile time
                    //then this function does nothing
                    //otherwise, it increment the internal ref count of the tensor
                    //in internal count is init to 0 by the tensor constructor
  }
}

void Context::initTensors(const TList &t_list) {
  for(auto t:t_list) {
    t->inFocus();
  }
}

void Context::deinitTensors(const TList &t_list) {
  for(auto t:t_list) {
    t->deFocus();
  }
}

void Context::dcrRefCount(TList t_list) {
  for(auto t:t_list) {
    t->dcrRef();
    if(t->getRef() < 1 && del_onsight) {
      delete t;
    }
  }
}

int Context::eval(void) {
  //unref2nullTensors();

  for(auto op:op_list) {
    initTensors(op->getInputs());
    initTensors(op->getOutputs());

    op->inFocus();
    op->compute();
    op->deFocus();

    deinitTensors(op->getInputs());
    deinitTensors(op->getOutputs());

    dcrRefCount(op->getInputs());

    op->dcrRef();
    if(op->getRef() < 1 && del_onsight) {
      delete op;
    }
  }

  return 0;
}
