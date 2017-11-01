#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>
#include "stdlib.h"
#include "uTensor_util.hpp"

class uTensor {
  virtual void inFocus(){};
  virtual void deFocus(){};

 public:
  virtual ~uTensor() = 0;
};

uTensor::~uTensor() {}
class TensorBase {
 public:
  std::vector<uint32_t> shape;
  void* data;
  uint32_t total_size;

  ~TensorBase() {
    if (data != nullptr) {
      std::cout << "i am tensorbase destructor " << std::endl;
      free(data);
      DEBUG("TensorBase memory freed..\r\n");
    }
  }
};

class Tensor : uTensor {
  virtual void* read(std::initializer_list<uint32_t> l) { return nullptr; }
  virtual void* write(size_t offset, size_t ele) { return nullptr; }

 protected:
  std::shared_ptr<TensorBase> s;  // short for states
 public:
  Tensor(void) { std::cout << "tensor constructor " << std::endl; }

  // returns how far a given dimension is apart
  size_t getStride(size_t dim_index) {
    unsigned int size_accm = 1;
    for (auto it = s->shape.begin() + dim_index + 1; it != s->shape.end();
         it++) {
      size_accm *= *it;
    }

    return (size_t)size_accm;
  }
  template <class T>
  void init(std::vector<uint32_t>& v) {
    std::cout << "initialize with type" << std::endl;
    s = std::make_shared<TensorBase>();
    s->total_size = 0;

    for (auto i : v) {
      s->shape.push_back(i);
      if (s->total_size == 0) {
        s->total_size = i;
      } else {
        s->total_size *= i;
      }
    }

    s->data = (void*)malloc(unit_size() * s->total_size);
    if (s->data == NULL)
      ERR_EXIT("ran out of memory for %lu malloc", unit_size() * s->total_size);
  }

  std::vector<uint32_t> getShape(void) { return s->shape; }

  uint32_t getSize(void) { return s->total_size; }

  virtual uint16_t unit_size(void) {}

  uint32_t getSize_in_bytes(void) { return s->total_size * unit_size(); }

  // returns the number of dimensions in the tensor
  size_t getDim(void) { return s->shape.size(); }

  template <class T>
  T* read(std::initializer_list<uint32_t> l) {
    return (T*)read(l);
  }

  ~Tensor() {
    s = nullptr;
    std::cout << "i am tensor destructor " << std::endl;
    DEBUG("Tensor Destructed\r\n");
  }
};

template <class T>
class RamTensor : public Tensor {
  // need deep copy
 public:
  RamTensor() : Tensor() {
    std::cout << "ramtensor " << std::endl;
    std::vector<uint32_t> v(3, 3);
    Tensor::init<T>(v);
    cursor = nullptr;
  }

  RamTensor(std::initializer_list<uint32_t> l) : Tensor() {
    std::cout << "ram con " << std::endl;
    std::vector<uint32_t> v;
    for (auto i : l) {
      v.push_back(i);
    }

    Tensor::init<T>(v);
  }

  RamTensor(std::vector<uint32_t>& v) : Tensor() {
    std::cout << "2 ram con " << std::endl;
    Tensor::init<T>(v);
  }

  // PRE:      l, initization list, specifying the element/dimension
  // POST:     When a degenerative index is supplied, the pointer
  //          lowest specified dimension is returned.
  //          Otherwise, return the pointer to the specific element.
  virtual void* read(std::initializer_list<uint32_t> l) override {
    size_t p_offset = 0;
    signed short current_dim = 0;
    for (auto i : l) {
      p_offset += i * getStride(current_dim);
      current_dim++;
    }

    // printf("p_offset: %d\r\n", p_offset);
    return (void*)((T*)s->data + p_offset);
  }

  /*  T* getPointer(std::vector<uint32_t> v) {
      size_t p_offset = 0;
      signed short current_dim = 0;
      for (auto i : v) {
        p_offset += i * getStride(current_dim);
        current_dim++;
      }

      printf("p_offset: %d\r\n", p_offset);

      return s->data + p_offset;
    }*/
  // virtual void* read(size_t offset, size_t ele) override{};
  virtual void* write(size_t offset, size_t ele) override{};
  virtual uint16_t unit_size(void) override { std::cout << "my unit size" << std::endl; return sizeof(T); }
  ~RamTensor() { std::cout << "i am ramtensor destructor" << std::endl; }

 private:
  T* cursor;
};

/*template <typename Tin, typename Tout>
Tensor<Tout> TensorCast(Tensor<Tin> input) {
  Tensor<Tout> output(input.getShape());
  Tin* inputPrt = input.getPointer({});
  Tout* outputPrt = output.getPointer({});

  for (uint32_t i = 0; i < input.getSize(); i++) {
    outputPrt[i] = static_cast<Tout>(inputPrt[i]);
  }

  return output;
}

template <typename T>
Tensor<T> TensorConstant(std::vector<uint32_t> shape, T c) {
  Tensor<T> output(shape);
  T* outPrt = output.getPointer({});

  for (uint32_t i = 0; i < output.getSize(); i++) {
    outPrt[i] = c;
  }

  return output;
}

template <typename T>
Tensor<T> TensorConstant(std::initializer_list<uint32_t> l, T c) {
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

  void computeOutputShape(void) {
    out_stride.clear();
    if (in_shape.empty()) ERR_EXIT("input shape not set");
    if (permute.empty() || permute.size() != in_shape.size())
      ERR_EXIT("invalid permute vector");

    for (auto&& curr_axis : permute) {
      out_shape.push_back(in_shape[curr_axis]);
    }
  }

  size_t evalStride(size_t dim_index, Shape s) {
    unsigned int size_accm = 1;
    for (auto it = s.begin() + dim_index + 1; it != s.end(); it++) {
      size_accm *= *it;
    }

    return (size_t)size_accm;
  }

  void computeInputStride(void) {
    in_stride.clear();
    for (uint32_t i = 0; i < in_shape.size(); i++) {
      in_stride.push_back(evalStride(i, in_shape));
    }
  }
  void computeOutputStride(void) {
    out_stride.clear();
    for (uint32_t i = 0; i < out_shape.size(); i++) {
      out_stride.push_back(evalStride(i, out_shape));
    }
  }

 public:
  permuteIndexTransform(Shape input_shape, std::vector<uint8_t> permute) {
    setInputShape(input_shape);
    setPermute(permute);
    apply();
  }

  std::vector<uint8_t> getPermute(void) { return permute; }
  void setPermute(std::vector<uint8_t>& _permute) {
    permute = _permute;
    depermute.resize(permute.size());
    uint8_t i = 0;
    for (auto a : permute) {
      depermute[a] = i;
      i++;
    }
  }

  void setInputShape(Shape s) { in_shape = s; }
  Shape getNewShape(void) { return out_shape; }

  void apply(void) {
    computeOutputShape();
    computeOutputStride();
    computeInputStride();
  }

  size_t operator[](const size_t index) {
    size_t out_index = 0;
    size_t rem = index;

    for (size_t curr_dim = 0; curr_dim < in_shape.size(); curr_dim++) {
      size_t curr_stride = in_stride[curr_dim];
      out_index += (rem / curr_stride) * out_stride[depermute[curr_dim]];
      rem = rem % curr_stride;
    }

    out_index += rem;

    return out_index;
  }

};

template <typename T>
void printDim(Tensor<T> t) {
  printf("Dimension: ");
  Shape s = t.getShape();
  for(auto d:s) {
    printf("[%lu] ", d);
  }
  printf("\r\n");
}

template <typename T>
void tensorChkAlloc(Tensor<T> &t, Shape dim) {
  if (t.getSize() == 0) {
    t = Tensor<T>(dim);
  } else if (t.getShape() != dim) {
    ERR_EXIT("Dim mismatched...\r\n");
  }
}*/
#endif
