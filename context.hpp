#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H

#include <unordered_map>

//TODO: how do we deal with dangling tensors?
//      only allow pushing for exact number of inputs
//      output reference count are initialized to 0, incremented only on input-push
//      outputs are allocated in ops
//      output lists can contain nullptr/empty-tensors
//      tensors can be all pointers here, but destructors has to set data to nullptr
//      push(op, input_t_list, output_t_list)  or  push(op, init-list, init-list)
//      TensorListModifierOp
class Context : uTensor {
protected:
  vector<Operator> op_list;
  std::unordered_map<Tensor*, int> tensor_lookup;

  void initOpTensors(vector<Tensor*> &t_list);
  void deinitTensors(vector<Tensor*> &t_list);
  void registerInputTensors(vector<Tensor*> &t_list);
  void registerOutputTensors(vector<Tensor*> &t_list);
  void decreRefCount(vector<Tensor*> &t_list);
  
  //void unref2nullTensors(vector<Tensor*> &t_list);
  //replace non-referenced output to null-tensors

public:
  void push(Operator op, TList &_inputs, TList &_outputs);
  int run(void);
};

void Context::push(Operator op, TList &_inputs, TList &_outputs) {
  if(op.getInputCount() != _inputs.size()) {
    ERR_EXIT("valid number of inputs\r\n");
  }
  if(op.getOutputCount() != _outputs.size()) {
    ERR_EXIT("valid number of output\r\n");
  }

  op.setInputs(_inputs);
  op.setInputs(_outputs);
  op_list.push_back(op);
  registerInputTensors(_inputs);
  registerOutputTensors(_outputs);

}


void Context::registerInputTensors(TList &t_list) {
  for(auto t:t_list) {
    auto ref_count = tensor_lookup.find(t);
    if(ref_count == tensor_lookup.end()) {
      tensor_lookup[t] = 1;
    } else {
      tensor_lookup[t]++;
    }
  }
}

void Context::registerOutputTensors(TList &t_list) {
  for(auto t:t_list) {
    auto ref_count = tensor_lookup.find(t);
    if(ref_count == tensor_lookup.end()) {
      tensor_lookup[t] = 0;
    }
  }
}


void Context::initOpTensors(vector<Tensor*> &t_list) {
  for(auto t:t_list) {
    t->inFocus();
  }
}

void Context::deinitTensors(vector<Tensor*> &t_list) {
  for(auto t:t_list) {
    t->deFocus();
  }
}

void Context::decreRefCount(vector<Tensor*> &t_list) {
  for(auto t:t_list) {
    tensor_lookup[t] = tensor_lookup[t] - 1;
    if(tensor_lookup[t] < 1) {
      t->~Tensor();
    }
}

int Context::run(void) {
  //unref2nullTensors();

  for(auto op:op_list) {
    initTensors(op.getInputs());
    initTensors(op.getOutputs());

    op.init();
    op.compute();
    op.deinit();

    deinitOpTensors(op.getInputs());
    deinitOpTensors(op.getOutputs());

    decreRefCount(op.getInputs());
  }
}

#endif // UTENSOR_CTX_H
