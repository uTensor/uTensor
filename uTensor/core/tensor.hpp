#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "uTensor/util/uTensor_util.hpp"
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <limits>

// enum class DType : char {
//   uint8,
//   int8,
//   uint16,
//   int32,
//   flt,
//   dbl,
// };

class Tensor;
class TensorIdxImporter;
typedef std::string TName;
typedef std::string OpName;
typedef std::vector<TName> TNameList;
typedef std::shared_ptr<Tensor> S_TENSOR;
typedef std::vector<S_TENSOR> S_TList;

class uTensor {
public:
 virtual void inFocus(){};
 virtual void deFocus(){};
 virtual std::string getName();
 virtual void setName(std::string _name);

 virtual ~uTensor() = 0;
private:
 std::string name;

};

//inline uTensor::~uTensor() {}
class TensorBase {
 public:
  std::vector<uint32_t> shape;
  void* data;
  uint32_t total_size;
  uint32_t cache_size;

  void initialize(std::vector<uint32_t>& vec);
  void allocate(uint8_t unit_size);

  ~TensorBase();
};

class Tensor : public uTensor {
  virtual void* read(size_t offset, size_t ele);
  virtual void* write(size_t offset, size_t ele);
  Tensor(const Tensor&);
  Tensor& operator=(const Tensor&);

 protected:
  std::shared_ptr<TensorBase> s;  // short for states
 public:
  Tensor();

  // returns how far a given dimension is apart
  size_t getStride(size_t dim_index); 

  virtual void init(std::vector<uint32_t>& v); 

  virtual void init(std::vector<uint32_t>& v, const void* data); 

  virtual void resize(std::vector<uint32_t> v); 

  std::vector<uint32_t> getShape(void); 

  uint32_t getSize(void); 

  virtual uint16_t unit_size(void); 

  uint32_t getSize_in_bytes(void); 

  // returns the number of dimensions in the tensor
  size_t getDim(void); 

  template <class T>
  const T* read(size_t offset, size_t ele) {
    return (const T*)read(offset, ele);
  }

  template <class T>
  T* write(size_t offset, size_t ele) {
    return (T*)write(offset, ele);
  }

  ~Tensor(); 
};

template<class T>
class BinaryTensor : public Tensor {
  public:
  BinaryTensor(std::vector<uint32_t> v, const T* g) : Tensor() {
    Tensor::init(v, g);
  }

  virtual uint16_t unit_size(void) override {
    return sizeof(T);
  }

  virtual void* read(size_t offset, size_t ele) override {
    if (ele > s->total_size) {
        ERR_EXIT("data overflow");
    }
    return (void *)((T*)s->data + offset);
  }

  virtual void* write(size_t offset, size_t ele) override {
    return nullptr;
  }
  ~BinaryTensor() {
    s->data = nullptr;
  }

 private:
  BinaryTensor(const BinaryTensor&);
  BinaryTensor& operator=(const BinaryTensor&);
};

template <class T>
class RamTensor : public Tensor {
  // need deep copy
 public:
  //RamTensor(TName _name) : Tensor(_name) {}
  RamTensor() {};

  RamTensor(std::initializer_list<uint32_t> l) {
    std::vector<uint32_t> v;
    for (auto i : l) {
      v.push_back(i);
    }

    Tensor::init(v);
  }

  RamTensor(std::vector<uint32_t> v) : Tensor() {
    Tensor::init(v);
  }

  // PRE:      l, initization list, specifying the element/dimension
  // POST:     When a degenerative index is supplied, the pointer
  //          lowest specified dimension is returned.
  //          Otherwise, return the pointer to the specific element.
  virtual void* read(size_t offset, size_t ele) override {
    if (ele > s->total_size) {
        ERR_EXIT("data overflow");
    }
    return (void *)((T*)s->data + offset);
  }
  virtual void* write(size_t offset, size_t ele) override {
    if (ele > s->total_size) {
        ERR_EXIT("data overflow");
    }
    return (void*)((T*)s->data + offset);
  }


  // virtual void* read(size_t offset, size_t ele) override{};
  virtual uint16_t unit_size(void) override {
    return sizeof(T);
  }
  ~RamTensor() {}
 private:
  RamTensor(const RamTensor&);
  RamTensor& operator=(const RamTensor&);

};

template <typename Tin, typename Tout>
Tensor* TensorCast(Tensor* input) {
  Tensor* output = new RamTensor<Tout>(input->getShape());
  const Tin* inputPrt = input->read<Tin>(0, 0);
  Tout* outputPrt = output->write<Tout>(0, 0);

  for (uint32_t i = 0; i < input->getSize(); i++) {
    outputPrt[i] = static_cast<Tout>(inputPrt[i]);
  }

  return output;
}

template <typename T>
Tensor* TensorConstant(std::vector<uint32_t> shape, T c) {
  Tensor* output = new RamTensor<T>(shape);
  T* outPrt = output->write<T>(0, 0);

  for (uint32_t i = 0; i < output->getSize(); i++) {
    outPrt[i] = c;
  }

  return output;
}

template <typename T>
Tensor* TensorConstant(std::initializer_list<uint32_t> l, T c) {
  std::vector<uint32_t> v;
  for (auto i : l) {
    v.push_back(i);
  }

  return TensorConstant<T>(v, c);
}

//
// permuteIndexTransform trans(inputTensor.getShape(), permute);
//
// Tensor<int> outputTensor(trans.getNewShape());  //of shape {100,40,10,10}
// size_t output_buffer_index = trans[input_buffer_index];

class permuteIndexTransform {
 private:
  std::vector<uint8_t> permute;
  std::vector<uint8_t> depermute;
  Shape in_shape;
  Shape in_stride;
  Shape out_shape;
  Shape out_stride;

  void computeOutputShape(void); 

  size_t evalStride(size_t dim_index, Shape s); 

  void computeInputStride(void); 
  void computeOutputStride(void); 

 public:
  permuteIndexTransform(Shape input_shape, std::vector<uint8_t> permute); 

  std::vector<uint8_t> getPermute(void); 
  void setPermute(std::vector<uint8_t>& _permute); 

  void setInputShape(Shape s); 
  Shape getNewShape(void); 

  void apply(void); 

  size_t operator[](const size_t index); 
};

template <typename T>
void printDim(Tensor* t) {
  printf("Dimension: ");
  Shape s = t->getShape();
  for (auto d : s) {
    printf("[%lu] ", d);
  }
  printf("\r\n");
}

template <typename T>
void tensorChkAlloc(Tensor** t, Shape dim) {
  if (*t && (*t)->getSize() == 0) {
    (*t)->init(dim);
  } else if (*t && (*t)->getShape() != dim) {
    ERR_EXIT("Dim mismatched...\r\n");
  } else if (*t == nullptr){
      *t = new RamTensor<T>(dim);
  }

}


//
// permuteIndexTransform trans(inputTensor.getShape(), permute);
//
// Tensor<int> outputTensor(trans.getNewShape());  //of shape {100,40,10,10}
// size_t output_buffer_index = trans[input_buffer_index];

class broadcastIndexTransform {
 private:
  Shape l_shape;
  Shape l_stride;
  Shape s_shape;
  Shape s_stride;
  bool swap_flag;

  size_t evalStride(size_t dim_index, Shape s); 

  void computeSStride(void); 
  void computeLStride(void); 

  void sortShape(Shape a, Shape b); 

  void checkShape(void); 


//b_shape being a smaller shape
 public:
  broadcastIndexTransform(Shape _l_shape, Shape _s_shape); 

  void apply(void); 

  Shape getOutputShape(void); 

  bool is_swaped(void); 

  size_t operator[](const size_t linear_index); 
};

#endif
