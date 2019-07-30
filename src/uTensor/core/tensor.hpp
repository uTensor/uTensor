#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "src/uTensor/util/uTensor_util.hpp"
#include "src/uTensor/util/utensor_string.hpp"
#include <initializer_list>
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
typedef utensor::string TName;
typedef utensor::string OpName;
typedef std::vector<TName> TNameList;
typedef std::shared_ptr<Tensor> S_TENSOR;
typedef std::vector<S_TENSOR> S_TList;

class TensorShape: public std::vector<uint32_t> {
    public:
        using std::vector<uint32_t>::vector;

};

class uTensor {
public:
 virtual void inFocus(){};
 virtual void deFocus(){};
 virtual const utensor::string& getName() const;
 virtual void setName(utensor::string _name);

 virtual ~uTensor() = 0;
private:
 utensor::string name;

};

//inline uTensor::~uTensor() {}
class TensorBase {
 public:
  TensorShape shape;
  void* data;
  uint32_t total_size;
  uint32_t cache_size;

  void initialize(const TensorShape& vec);
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

  virtual void init(const TensorShape& v); 

  virtual void init(const TensorShape& v, const void* data); 

  virtual void resize(const TensorShape& v); 

  const TensorShape& getShape(void) const; 

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

  virtual ~Tensor(); 
};

template<class T>
class BinaryTensor : public Tensor {
  public:
  BinaryTensor(const TensorShape& v, const T* g) : Tensor() {
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
  
  virtual ~BinaryTensor() {
    s->data = nullptr;
  }

 private:
  BinaryTensor(const BinaryTensor&);
  BinaryTensor& operator=(const BinaryTensor&);
};

template <class T>
class RamTensor : public Tensor {
  // need deep copy
 protected:
    using Tensor::s;
 public:
  //RamTensor(TName _name) : Tensor(_name) {}
  RamTensor() {};

  RamTensor(std::initializer_list<uint32_t> l) {
    TensorShape v;
    for (auto i : l) {
      v.push_back(i);
    }

    Tensor::init(v);
  }

  RamTensor(const TensorShape& v) : Tensor() {
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
  
  virtual ~RamTensor() {}
 private:
  RamTensor(const RamTensor&);
  RamTensor& operator=(const RamTensor&);

};


// application owns the pointer
// the printer may be modified outside of the application
template <class T>
class WrappedRamTensor : public RamTensor<T> {

 protected:
  using RamTensor<T>::s;

 public:
  WrappedRamTensor() {};

  WrappedRamTensor(std::initializer_list<uint32_t> l, T* ptr) {
    TensorShape v;
    for (auto i : l) {
      v.push_back(i);
    }

    void* data = (void *) ptr;
    
    Tensor::init(v, data);
  }

  void setPointer(void* ptr) {
    s->data = ptr;
  }

  T* getPointer(void) {
    return (T*) s->data;
  }

  virtual
  ~WrappedRamTensor() {
    s->data = nullptr;
  }

 private:
  WrappedRamTensor(const WrappedRamTensor&);
  WrappedRamTensor& operator=(const WrappedRamTensor&);

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
Tensor* TensorConstant(const TensorShape& shape, T c) {
  Tensor* output = new RamTensor<T>(shape);
  T* outPrt = output->write<T>(0, 0);

  for (uint32_t i = 0; i < output->getSize(); i++) {
    outPrt[i] = c;
  }

  return output;
}

template <typename T>
Tensor* TensorConstant(std::initializer_list<uint32_t> l, T c) {
  TensorShape v;
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
  TensorShape in_shape;
  TensorShape in_stride;
  TensorShape out_shape;
  TensorShape out_stride;

  void computeOutputShape(void); 

  size_t evalStride(size_t dim_index, const TensorShape& s); 

  void computeInputStride(void); 
  void computeOutputStride(void); 

 public:
  permuteIndexTransform(const TensorShape& input_shape, const std::vector<uint8_t>& permute); 

  const std::vector<uint8_t>& getPermute(void) const;
  void setPermute(const std::vector<uint8_t>& _permute); 

  void setInputShape(const TensorShape& s); 
  TensorShape getNewShape(void); 

  void apply(void); 

  size_t operator[](const size_t index); 
};

template <typename T>
void printDim(Tensor* t) {
  printf("Dimension: ");
  TensorShape s = t->getShape();
  for (auto d : s) {
    printf("[%lu] ", d);
  }
  printf("\r\n");
}

template <typename T>
void tensorChkAlloc(Tensor** t, const TensorShape& dim) {
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
  TensorShape l_shape;
  TensorShape l_stride;
  TensorShape s_shape;
  TensorShape s_stride;
  bool swap_flag;

  size_t evalStride(size_t dim_index, const TensorShape& s); 

  void computeSStride(void); 
  void computeLStride(void); 

  void sortShape(const TensorShape& a, const TensorShape& b); 

  void checkShape(void); 


//b_shape being a smaller shape
 public:
  broadcastIndexTransform(const TensorShape& _l_shape, const TensorShape& _s_shape); 

  void apply(void); 

  const TensorShape& getOutputShape(void) const;

  bool is_swaped(void); 

  size_t operator[](const size_t linear_index); 
};

#endif
