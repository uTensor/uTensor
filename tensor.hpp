#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include <initializer_list>
#include <memory> // shared_ptr, make_shared
#include <uTensor_util.hpp>
#include <vector>
#include "mbed.h" // Serial, AnalogIN
#include "stdlib.h"

template <typename U>
static U _reduce_mul(vector<U> v) {
  U acc = 1;
  for (auto d : v) {
    acc *= d;
  }
  return acc;
}

template <class U>
class TensorBase {
 public:
  vector<uint32_t> shape;
  U* data;
  uint32_t total_size;

  TensorBase(vector<uint32_t> v) {
    init(v);
  }

  TensorBase(initializer_list<uint32_t> l) {
    vector<uint32_t> v;
    v.reserve(l.size());
    for (auto d : l) {
      v.push_back(d);
    }
    init(v);
  }

  ~TensorBase() {
    DEBUG("TensorBase destruction..\r\n");
    free(data);
  }
private:
  void init(vector<uint32_t> v) {
    shape = v;
    size_t num_elems = static_cast<size_t>(_reduce_mul(v));
    data = (U*) malloc(sizeof(U)*num_elems);
    total_size = _reduce_mul(v);
  }
};

template <class T>
class Tensor {
  std::shared_ptr<TensorBase<T>> s;  // short for states

 public:
  Tensor(void) {
    s = make_shared<TensorBase<T>>({0});
  }
  Tensor(initializer_list<uint32_t> l){
    s = make_shared<TensorBase<T>>(l);
  }
  Tensor(vector<uint32_t> v) {
    s = make_shared<TensorBase<T>>(v);
  }

  Tensor(initializer_list<T> data, initializer_list<uint32_t> shape) {
    s = make_shared<TensorBase<T>>(shape);
    // populate data
    size_t offset = 0;
    for (auto d : data) {
      *(s->data+offset) = d;
      offset++;
    }
  }

  Tensor(vector<T> data, initializer_list<uint32_t> shape) {
    s = make_shared<TensorBase<T>>(shape);
    size_t offset = 0;
    for (auto d : data) {
      *(s->data+offset) = d;
      offset++;
    }
  }

  // returns how far a given dimension is apart
  size_t getStride(size_t dim_index) {
    unsigned int size_accm = 1;
    for (auto it = s->shape.begin() + dim_index + 1; it != s->shape.end();
         it++) {
      size_accm *= *it;
    }

    return (size_t)size_accm;
  }

  // PRE:      l, initization list, specifying the element/dimension
  // POST:     When a degenerative index is supplied, the pointer
  //          lowest specified dimension is returned.
  //          Otherwise, return the pointer to the specific element.
  T* getPointer(initializer_list<size_t> l) {
    size_t p_offset = 0;
    signed short current_dim = 0;
    for (auto i : l) {
      p_offset += i * getStride(current_dim);
      current_dim++;
    }

    // printf("p_offset: %d\r\n", p_offset);
    return s->data + p_offset;
  }

  T* getPointer(vector<uint32_t> v) {
    size_t p_offset = 0;
    signed short current_dim = 0;
    for (auto i : v) {
      p_offset += i * getStride(current_dim);
      current_dim++;
    }

    printf("p_offset: %d\r\n", p_offset);

    return s->data + p_offset;
  }

  vector<uint32_t> getShape(void) { return s->shape; }

  uint32_t getSize(void) { return s->total_size; }

  uint16_t unit_size(void) { return sizeof(T); }

  uint32_t getSize_in_bytes(void) { return s->total_size * unit_size(); }

  // returns the number of dimensions in the tensor
  size_t getDim(void) { return s->shape.size(); }

  ~Tensor() {
    // if(data != NULL)
    //     free(data);

    DEBUG("Tensor Destructed\r\n");
  }
};

template <typename Tin, typename Tout>
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
Tensor<T> TensorConstant(vector<uint32_t> shape, T c) {
  Tensor<T> output(shape);
  T* outPrt = output.getPointer({});

  for (uint32_t i = 0; i < output.getSize(); i++) {
    outPrt[i] = c;
  }

  return output;
}

template <typename T>
Tensor<T> TensorConstant(initializer_list<uint32_t> l, T c) {
  vector<uint32_t> v;
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
  vector<uint8_t> permute;
  vector<uint8_t> depermute;
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
  permuteIndexTransform(Shape input_shape, vector<uint8_t> permute) {
    setInputShape(input_shape);
    setPermute(permute);
    apply();
  }

  vector<uint8_t> getPermute(void) { return permute; }
  void setPermute(vector<uint8_t>& _permute) {
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

#endif
