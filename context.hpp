#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H

#include <memory>
#include <unordered_map>
#include <initializer_list>
#include "uTensorBase.hpp"
#include "stdio.h"
//#include <list>

class Ref_Record {
public:
  uint8_t count;
  bool allow_incr;
  S_TENSOR sptr;

  Ref_Record() {
    count = 0;
    allow_incr = true;
    sptr.reset();   
  }
};

class Context : public uTensor {
protected:
  std::vector<Operator*> op_list;
  bool del_onsight;

  std::unordered_map<TName, Ref_Record> rTable;  //all tensors alive  //kill all unused if malloc failed?
  //uint32_t m_size; //remaining memory size
  //void registerTensor(Tensor* t);
  //void gc(void); //garbage collector, delete any tracked unreferenced tensor

  void initTensors(const S_TList &t_list);
  void deinitTensors(const S_TList &t_list);

  void incrTNameListRef(const TNameList &t_list);
  void dcrListRef(S_TList t_list);
  void delTensor(TName t);
  //uint16_t incrRef(std::shared_ptr<Tensor> sptr);
  uint8_t dcrRef(TName name);
  bool isTracked(TName name);
  //uint16_t getRef();

public:
  S_TENSOR add(Tensor* t, uint8_t init_count = 0);
  void push(Operator *op, TNameList &_inputs, TNameList &_outputs);
  void push(Operator *op, std::initializer_list<TName> _inputs, std::initializer_list<TName> _outputs);
  int eval(void);

  Context() {
    del_onsight = true;
  }
};



#endif // UTENSOR_CTX_H
