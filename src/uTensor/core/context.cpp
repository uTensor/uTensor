#include "src/uTensor/core/context.hpp"

S_TENSOR Context::add(Tensor* t, TName _name, uint8_t init_count) {
  if(t == nullptr) { ERR_EXIT("null pointer tensor"); }
  if(rTable.find(_name) != rTable.end()) {
    ERR_EXIT("tensor with name \"%d\" address already exist in rTable", t->getName().get_value());
  }

  t->setName(_name);

  Ref_Record record;

  if(init_count != 0) {
    record.count = init_count;
    record.allow_incr = false;
  }

  record.ptr = t;

  rTable[_name] = record;

  return t;
}

S_TENSOR Context::get(TName const &t_name) {
  if(rTable.find(t_name) == rTable.end()) ERR_EXIT("No tensor with name: %d", t_name.get_value());
  return rTable[t_name].ptr;
}

Operator* Context::registerOpTable(std::function<void*(void)> func, TName _name) {
  Operator* op;
  //empty static op tensor list
  if(opTable.find(_name) == opTable.end()) {
    op = (Operator*) func();
    op->setName(_name);
    opTable[_name] = op;
  } else {
    op = opTable[_name];
  }
  
  return op;
}

void Context::push(Operator* op, TName _name, TNameList &_inputs, TNameList &_outputs) {
  Context::push(op, _inputs, _outputs);
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
    _inputs.push_back(r.ptr);
  }

  S_TList _outputs;
  for(auto out:out_names) {
    if(rTable.find(out) == rTable.end()) { ERR_EXIT("Tensor \"%d\" not found", out.get_value()); }
    Ref_Record r = rTable[out];
    _outputs.push_back(r.ptr);
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
  delete record.ptr;
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
  
  //FIXME: delete ops referenced by the opTable
  return (uint32_t) nlist.size();
}

