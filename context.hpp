#ifndef UTENSOR_CTX_H
#define UTENSOR_CTX_H

#include "uTensorBase.hpp"
#include "stdio.h"

//#include <list>

//TODO: how do we deal with dangling tensors?
//      only allow pushing for exact number of inputs
//      output reference count are initialized to 0, incremented only on input-push
//      outputs are allocated in ops
//      output lists can contain nullptr/empty-tensors
//      tensors can be all pointers here, but destructors has to set data to nullptr
//      push(op, input_t_list, output_t_list)  or  push(op, init-list, init-list)
//      TensorListModifierOp
class Context : public uTensor {
protected:
  std::vector<Operator*> op_list;
  bool del_onsight;
  //std::unordered_map<Tensor*> TensorList;  //all tensors alive  //kill all unused if malloc failed?
  //uint32_t m_size; //remaining memory size
  //void registerTensor(Tensor* t);
  //void gc(void); //garbage collector, delete any tracked unreferenced tensor

  void initTensors(const TList &t_list);
  void deinitTensors(const TList &t_list);
  void updateInputTensorRef(const TList &t_list);
  void dcrRefCount(TList t_list);

public:
  void push(Operator *op, TList &_inputs, TList &_outputs);
  int eval(void);

  Context() {
    del_onsight = true;
  }
};


#endif // UTENSOR_CTX_H
