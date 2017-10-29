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
  uint8_t tmp_input_count;
  uint8_t tmp_output_count;
  vector<TensorBase*> tmp_input_list;
  vector<TensorBase*> tmp_output_list;

  void runOp(Operator &op);
  void initOpTensors(Operator &op);
  void deinitOpTensors(Operator &op);
  void injectOp(void);

public:
  Context() {
    tmp_input_count = 0;
    tmp_output_count = 0;
  }
  void addOp(Operator op);
  void addInputs(Operator op);
  void push(void);
  vector<TensorBase*> Context::getOutputs(void);
  int run(void);
};


void Context::addOp(Operator &op) {
  if(tmp_input_count != 0) {
    ERR_EXIT("valid number of inputs\r\n");
  }
  if(tmp_output_count != 0) {
    ERR_EXIT("valid number of outputs\r\n");
  }

  op_list.push_back(op);
  tmp_input_count = op.getInputCount();
  tmp_output_count = op.getOutputCount();
}

void Context::addInputs(vector<TensorBase*> t_list) {
  int tmp_input_count = tmp_input_count - t_list.size();
  if(tmp_input_count < 0) ERR_EXIT("supplied too many inputs");
  tmp_input_list.insert(tmp_input_list.end(), t_list.begin(), t_list.end());

  for(auto t:t_list) {
    auto ref_count = tensor_refs.find(t);
    if(ref_count == tensor_refs.end()) {
      tensor_refs[t] = 1;
    } else {
      tensor_refs[t]++;
    }
  }

}

void Context::push(void) {
  if(tmp_input_count != 0 &&
      tmp_output_count != 0) {
    ERR_EXIT("valid number of inputs/outputs\r\n");
  }

  auto op = op_list.back();  
  op.setInputs(tmp_input_list);
  op.setOutputs(tmp_output_list);

  tmp_input_list.empty();
  tmp_output_list.empty();
  tmp_input_count = 0;
  tmp_output_count = 0;
}

vector<TensorBase*> Context::getOutputs(void) {

}

void Context::runOp(Operator &op) {

}

int Context::run(void) {
  tensorCleanup();

  for(auto op:op_list) {
    initOpTensors(op.getInputs());
    initOpTensors(op.getOutputs());

    runOp(op);

    deinitOpTensors(op.getInputs());
    deinitOpTensors(op.getOutputs());

    decreRefCount(op.getInputs());
    tensorCleanup();
  }
}

#endif // UTENSOR_CTX_H
