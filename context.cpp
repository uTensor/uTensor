#include "context.hpp"

TENSOR Context::add(Tensor* t, uint8_t init_count) {
  if(rTable.find(t) != rTable.end()) {
    ERR_EXIT("tensor pointer address already exist in rTable");
  }

  S_TENSOR _sptr(t);

  Ref_Record record;

  if(init_count != 0) {
    record.count = init_count;
    record.allow_incr = false;
  }
  record.sptr = _sptr;

  rTable[t] = record;

  TENSOR wptr = _sptr;

  return wptr;
}


void Context::push(Operator *op, TList &_inputs, TList &_outputs) {
  //error checking in the Op class
  op->setInputs(_inputs);
  op->setOutputs(_outputs);
  op_list.push_back(op);
  incrTListRef(_inputs);

}

void Context::push(Operator *op, std::initializer_list<TENSOR> _inputs, std::initializer_list<TENSOR> _outputs) {
  TList inputs;
  TList outputs;

  for(auto i:_inputs) {
    inputs.push_back(i);
  }

  for(auto o:_outputs) {
    outputs.push_back(o);
  }

  push(op, inputs, outputs);
}

void Context::incrTListRef(const TList &t_list) {
  for(auto t:t_list) {
    Tensor* ptr = t.lock().get();
    if(rTable.find(ptr) == rTable.end()) {
      ERR_EXIT("tensor not registered");
    }

    Ref_Record record = rTable[ptr];
    if(record.allow_incr) {
      record.count++;
      rTable[ptr] = record;
    }
    
      //if an initial ref value is supplied to the tensor at compile time
                    //then this function does nothing
                    //otherwise, it increment the  ref count of the tensor
                    //count is init to 0 by the record constructor
  }
}

void Context::initTensors(const S_TList &t_list) {
  for(auto t:t_list) {
    t->inFocus();
  }
}

void Context::deinitTensors(const S_TList &t_list) {
  for(auto t:t_list) {
    t->deFocus();
  }
}

void Context::delTensor(Tensor* t) {
  Ref_Record record = rTable[t];
  record.sptr.reset();
  rTable.erase(t);
}

void Context::dcrListRef(S_TList t_list) {
  for(auto t:t_list) {
    if(dcrRef(t.get()) < 1) {
      delTensor(t.get());
    }
  }
}

uint8_t Context::dcrRef(Tensor* t) {
  if(!isTracked(t)) {
    ERR_EXIT("Tensor not registered");
  }

  Ref_Record record = rTable[t];
  if(record.count > 0) record.count -= 1;
  rTable[t] = record;

  return record.count;
}

bool Context::isTracked(Tensor* t) {
  return (rTable.find(t) != rTable.end());
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

    dcrListRef(op->getInputs());

    delete op;

  }

  op_list.clear();

  return 0;
}