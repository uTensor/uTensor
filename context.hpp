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

  std::unordered_map<Tensor*, Ref_Record> rTable;  //all tensors alive  //kill all unused if malloc failed?
  //uint32_t m_size; //remaining memory size
  //void registerTensor(Tensor* t);
  //void gc(void); //garbage collector, delete any tracked unreferenced tensor

  void initTensors(const S_TList &t_list);
  void deinitTensors(const S_TList &t_list);

  void incrTListRef(const TList &t_list);
  void dcrListRef(S_TList t_list);
  void delTensor(Tensor* t);
  //uint16_t incrRef(std::shared_ptr<Tensor> sptr);
  uint8_t dcrRef(Tensor* t);
  bool isTracked(Tensor* t);
  //uint16_t getRef();

public:
  TENSOR add(Tensor* t, uint8_t init_count = 0);
  void push(Operator *op, TList &_inputs, TList &_outputs);
  void push(Operator *op, std::initializer_list<TENSOR> _inputs, std::initializer_list<TENSOR> _outputs);
  int eval(void);

  Context() {
    del_onsight = true;
  }
};



#endif // UTENSOR_CTX_H
