#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "memoryManagementInterface.hpp"
#include "tensorBase.hpp"
#include "utensor_string.hpp"

namespace uTensor {
// Tensors also appear on the same heap as the Tensor metadata. This way we can
// move tensors around and delete them without affecting user code
// template <typename Allocator=utensor::DefaultTensorMetaDataAllocator>
class Tensor : public Handle {
  // private:
  //    // Cannot copy Tensors, must pass by reference
  //    Tensor(const Tensor& that);
  //enum Type : uint8_t { TENSOR_IFC_PTR, TENSOR_HANDLE_PTR } type;

 public:
  TensorInterface* operator->();
  Tensor(TensorInterface* ptr);
  // Add some bits to make the interface nicer to the user

  // Force everything to be on the utensor allocator
  void* operator new(size_t sz);
  void operator delete(void* p);

  // KEY BIT
  friend class AllocatorInterface;
};
/*
  class Tensor {
    enum Type : uint8_t { TENSOR_BASE_PTR, TENSOR_HANDLE_PTR } type;
    union {
        TensorBase* tb;
        Tensor* tp;
    };

    public:
        Tensor(TensorBase* tb) : type(Type::TENSOR_BASE_PTR), tb(tb) {}
        // Slightly different behavior from regular copy
        Tensor(const Tensor& tp) : type(Type::TENSOR_HANDLE_PTR), tp(const_cast<Tensor*>(&tp)) {}
        //Tensor& operator= (const Tensor& tp){ type=Type::TENSOR_HANDLE_PTR; tp = const_cast<Tensor*>(&tp); return *this; }
        Tensor& operator= (Tensor& tp){ type = Type::TENSOR_HANDLE_PTR; this->tp = &tp; return *this; }
        Tensor& operator= (TensorBase* tb){ type=Type::TENSOR_BASE_PTR; tb = tb; return *this; }
        //Do move semantics as well
        Tensor(Tensor&& that) : type(that.type) {
            if(type == Type::TENSOR_BASE_PTR)
                tb = that.tb;
            else
                tp = that.tp;
        }
        Tensor& operator=(Tensor&& that) {
            if(this != &that){
                this->type = that.type;
                if(type == Type::TENSOR_BASE_PTR){
                    tb = that.tb;
                    that.tb = nullptr;
                }
                else{
                    tp = that.tp;
                    that.tb = nullptr;
                }

            }
            return *this;
        }

        void read(int i) {
            cout << "In Tensor read" << endl;
            if(type == Type::TENSOR_BASE_PTR)
                tb->read(i);
            else
                tp->read(i);
        }

};
*/

// Same as Named Tensor but not registered in the context class
struct SimpleNamedTensor {
 public:
  const uTensor::string& name;  // Fixed
  Tensor& tensor;               // Modifiable

  SimpleNamedTensor(const uTensor::string& name, Tensor& tensor);
};
}  // namespace uTensor
#endif
