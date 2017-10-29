#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H

#include <unordered_map>
#include <typeinfo>
#include "tensor.hpp"

typedef long long TensorPtr;

class uTensor {
  virtual void init(Context ctx) {};
  virtual void inFocus() {};
  virtual void deFocus() {};
  virtual void finalize() {};
  virtual ~uTensor() = 0;
}

//isType() https://stackoverflow.com/questions/9974596/how-to-check-whether-two-pointers-point-to-the-same-object-or-not
//double dispatch

//new vs stack
class Operator : uTensor {
protected:
  //setup input/output info in derived constructors
  vector<TensorBase*> inputs;
  vector<DType> dtype_in;
  vector<TensorBase*> outputs;
  vector<DType> dtype_out;
public:
  virtual void compute() = 0;

  void setInputs(vector<TensorBase*> &_inputs) {
    if(_inputs.size() != inputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < input.size(); i++) {
      if(dtype_in[i] == inputs.getType()) {
        input[i] = _inputs[i];
      } else {
        ERR_EXIT("Tensor Type mismatched...");
      }
    }
  }

  void setOutputs(vector<TensorBase*> &_outputs) {
    if(_outputs.size() != outputs.size()) ERR_EXIT("Input Tensor list mismatched...");

    for(uint8_t i = 0; i < output.size(); i++) {
      output[i] = _output[i]
      if(_output[i] == nullptr) continue;
      if(dtype_out[i].getType() != output[i].getType()) ERR_EXIT("Tensor Type mismatched...");
    }
  }

  vector<TensorBase*> getInputs(void) {
    return inputs;
  }

  vector<TensorBase*> getOutputs(void) {
    return outputs;
  }
};

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
  std::unordered_map<TensorBase*, int> tensor_refs;

  void initOpTensors(vector<TensorBase*> &t_list);
  void deinitTensors(vector<TensorBase*> &t_list);
  void registerInputTensors(vector<TensorBase*> &t_list);
  void registerOutputTensors(vector<TensorBase*> &t_list);
  void decreRefCount(vector<TensorBase*> &t_list);
  
  //void unref2nullTensors(vector<TensorBase*> &t_list);
  //replace non-referenced output to null-tensors

public:
  Context() {
    tmp_input_count = 0;
    tmp_output_count = 0;
  }

  void push(Operator op, vector<TensorBase*> _inputs, vector<TensorBase*> _outputs);
  int run(void);
};

void push(Operator op, vector<TensorBase*> _inputs, vector<TensorBase*> _outputs) {
  if(op.getInputCount() != _inputs.size()) {
    ERR_EXIT("valid number of inputs\r\n");
  }
  if(op.getOutputCount() != _outputs.size()) {
    ERR_EXIT("valid number of output\r\n");
  }

  op_list.push_back(op);
  registerInputTensors(_inputs);
  registerOutputTensors(_outputs);

}


void Context::registerInputTensors(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    auto ref_count = tensor_refs.find(t);
    if(ref_count == tensor_refs.end()) {
      tensor_refs[t] = 1;
    } else {
      tensor_refs[t]++;
    }
  }
}

void Context::registerOutputTensors(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    auto ref_count = tensor_refs.find(t);
    if(ref_count == tensor_refs.end()) {
      tensor_refs[t] = 0;
    }
  }
}


void Context::initOpTensors(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    t.inFocus();
  }
}

void Context::deinitTensors(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    t.deFocus();
  }
}

void Context::deinitTensors(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    t.deFocus();
  }
}

void Context::decreRefCount(vector<TensorBase*> &t_list) {
  for(auto t:t_list) {
    tensor_refs[t] = tensor_refs[t] - 1;
    if(tensor_refs[t] < 1) {
      t.~Tensor();
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
