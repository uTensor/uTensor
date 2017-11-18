#include "context.hpp"

S_TENSOR Context::add(Tensor* t, uint8_t init_count) {
  if(t == nullptr) { ERR_EXIT("null pointer tensor"); }
  if(rTable.find(t->getName()) != rTable.end()) {
    ERR_EXIT("tensor pointer address already exist in rTable");
  }

  S_TENSOR _sptr(t);

  Ref_Record record;

  if(init_count != 0) {
    record.count = init_count;
    record.allow_incr = false;
  }

  record.sptr = _sptr;

  rTable[t->getName()] = record;

  return _sptr;
}


void Context::push(Operator *op, TNameList &in_names, TNameList &out_names) {
  //error checking in the Op class
  S_TList _inputs;
  for(auto in:in_names) {
    if(rTable.find(in) == rTable.end()) { ERR_EXIT("Tensor \"%s\" not found", in.c_str()); }
    Ref_Record r = rTable[in];
    _inputs.push_back(r.sptr);
  }

  S_TList _outputs;
  for(auto out:out_names) {
    if(rTable.find(out) == rTable.end()) { ERR_EXIT("Tensor \"%s\" not found", out.c_str()); }
    Ref_Record r = rTable[out];
    _outputs.push_back(r.sptr);
  }

  op->setInputs(_inputs);
  op->setOutputs(_outputs);
  op_list.push_back(op);
  incrTNameListRef(in_names);

}

void Context::push(Operator *op, std::initializer_list<TName> _inputs, std::initializer_list<TName> _outputs) {
  TNameList inputs;
  TNameList outputs;

  for(auto i:_inputs) {
    inputs.push_back(i);
  }

  for(auto o:_outputs) {
    outputs.push_back(o);
  }

  push(op, inputs, outputs);
}

void Context::incrTNameListRef(const TNameList &t_list) {
  for(auto t_name:t_list) {
    if(rTable.find(t_name) == rTable.end()) {
      ERR_EXIT("tensor not registered");
    }

    Ref_Record record = rTable[t_name];
    if(record.allow_incr) {
      record.count++;
      rTable[t_name] = record;
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

void Context::delTensor(TName t_name) {
  Ref_Record record = rTable[t_name];
  record.sptr.reset();
  rTable.erase(t_name);
}

void Context::dcrListRef(S_TList t_list) {
  for(auto t:t_list) {
    if(dcrRef(t->getName()) < 1) {
      delTensor(t->getName());
    }
  }
}

uint8_t Context::dcrRef(TName t_name) {
  if(!isTracked(t_name)) {
    ERR_EXIT("Tensor not registered");
  }

  Ref_Record record = rTable[t_name];
  if(record.count > 0) record.count -= 1;
  rTable[t_name] = record;

  return record.count;
}

bool Context::isTracked(TName t_name) {
  return (rTable.find(t_name) != rTable.end());
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
