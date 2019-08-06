#include "src/uTensor/core/context.hpp"

S_TENSOR Context::add_static(std::function<void*(void)> func, TName _name) {
  return addCached(func, _name, 1, true);
}

S_TENSOR Context::addCached(std::function<void*(void)> func, TName _name, uint8_t init_count, bool _is_static) {
  Tensor* t;
  if(rTable.find(_name) == rTable.end()) {
    t = (Tensor*) func();
    add(t, _name);
  }

  Ref_Record record = rTable[_name];
  record.is_static = _is_static;
  record.is_cacheable = true;
  if(init_count > 0) {
    record.count = init_count;
    record.allow_incr = false;
  }
  if(record.count < 1 && record.is_static) {
    record.count = 1;
  }
  rTable[_name] = record;

  return record.sptr;
}

S_TENSOR Context::add(Tensor* t, TName _name, uint8_t init_count) {
  if(t == nullptr) { ERR_EXIT("null pointer tensor"); }
  if(rTable.find(_name) != rTable.end()) {
    ERR_EXIT("tensor with name \"%d\" address already exist in rTable", t->getName().get_value());
  }

  S_TENSOR _sptr(t);
  t->setName(_name);

  Ref_Record record;

  if(init_count != 0) {
    record.count = init_count;
    record.allow_incr = false;
  }

  record.sptr = _sptr;

  rTable[_name] = record;

  return _sptr;
}

S_TENSOR Context::get(TName const &t_name) {
  if(rTable.find(t_name) == rTable.end()) ERR_EXIT("No tensor with name: %d", t_name.get_value());
  return rTable[t_name].sptr;
}

Operator* Context::registerOpTable(std::function<void*(void)> func, TName _name) {
  Operator* op;
  //empty static op tensor list
  if(opTable.find(_name) == opTable.end()) {
    op = (Operator*) func();
    op->setName(_name);
  } else {
    op = opTable[_name];
  }
  
  return op;
}

void Context::push_static(std::function<void*(void)> func, TName _name, TNameList &_inputs, TNameList &_outputs, bool is_static) {
  push(registerOpTable(func, _name), _inputs, _outputs);
}
void Context::push_static(std::function<void*(void)> func, TName _name, std::initializer_list<TName> _inputs, std::initializer_list<TName> _outputs, bool is_static) {
  push(registerOpTable(func, _name), _inputs, _outputs);
}

void Context::push(Operator* op, std::initializer_list<TName> _inputs, std::initializer_list<TName> _outputs) {
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

void Context::push(Operator* op, TNameList &in_names, TNameList &out_names) {
  //error checking in the Op class
  S_TList _inputs;
  for(auto in:in_names) {
    if(rTable.find(in) == rTable.end()) { ERR_EXIT("Tensor \"%d\" not found", in.get_value()); }
    Ref_Record r = rTable[in];
    _inputs.push_back(r.sptr);
  }

  S_TList _outputs;
  for(auto out:out_names) {
    if(rTable.find(out) == rTable.end()) { ERR_EXIT("Tensor \"%d\" not found", out.get_value()); }
    Ref_Record r = rTable[out];
    _outputs.push_back(r.sptr);
  }

  op->setInputs(_inputs);
  op->setOutputs(_outputs);
  op_list.push_back(op);
  incrTNameListRef(in_names);

}

void Context::incrTNameListRef(const TNameList &t_list) {
  for(auto t_name:t_list) {
    if(rTable.find(t_name) == rTable.end()) {
      ERR_EXIT("tensor not registered");
    }

    Ref_Record record = rTable[t_name];
    if(record.allow_incr && !record.is_static) {
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
  if(record.count > 0 && !record.is_static) record.count -= 1;
  rTable[t_name] = record;

  return record.count;
}

bool Context::isTracked(TName t_name) {
  return (rTable.find(t_name) != rTable.end());
}

void Context::cleanUpOp(Operator* op) {
  if(opTable.find(op->getName()) == opTable.end()) {
    delete op;
  } else {
    op->empty();
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

    dcrListRef(op->getInputs());


    cleanUpOp(op);

  }

  op_list.clear();

  return 0;
}

uint32_t Context::gc(void) {
  TNameList nlist;
  ///NT: TODO: implement cache policy here

  for ( auto it : rTable) {
    Ref_Record r = it.second;
    if(r.count < 1) {
      nlist.push_back(it.first);
    }
  }

  for(auto name:nlist) {
    delTensor(name);
  }
  
  return (uint32_t) nlist.size();
}

