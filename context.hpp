#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H

//#include <list>

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
  bool del_after_op;
  //std::unordered_map<Tensor*> TensorList;  //all tensors alive  //kill all unused if malloc failed?
  //uint32_t m_size; //remaining memory size
  //void registerTensor(Tensor* t);
  //void gc(void); //garbage collector, delete any tracked unreferenced tensor

  void initOpTensors(TList &t_list);
  void deinitTensors(TList &t_list);
  void updateInputTensorRef(TList &t_list);
  void dcrRefCount(TList &t_list);

public:
  void push(Operator op, TList &_inputs, TList &_outputs);
  int run(void);
};

Context() {
  del_onsight = true;
}

void Context::push(Operator op, TList &_inputs, TList &_outputs) {
  if(op.getInputCount() != _inputs.size()) {
    ERR_EXIT("valid number of inputs\r\n");
  }
  if(op.getOutputCount() != _outputs.size()) {
    ERR_EXIT("valid number of output\r\n");
  }

  op.setInputs(_inputs);
  op.setOutputs(_outputs);
  op_list.push_back(op);
  updateInputTensorRef(_inputs);

}

void Context::updateInputTensorRef(TList &t_list) {
  for(auto t:t_list) {
    t->incrRef();  //if an initial ref value is supplied to the tensor at compile time
                    //then this function does nothing
                    //otherwise, it increment the internal ref count of the tensor
                    //in internal count is init to 0 by the tensor constructor
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

void Context::dcrRefCount(vector<Tensor*> &t_list) {
  for(auto t:t_list) {
    t->dcrRef();
    if(t->getRef() < 1 && del_onsight) {
      delete t;
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
