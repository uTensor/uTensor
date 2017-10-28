#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include <initializer_list>
#include <memory>
#include <uTensor_util.hpp>
#include <vector>
#include "mbed.h"
#include "stdlib.h"

template <class U>
class TensorBase {
 public:
  vector<uint32_t> shape;
  U* data;
  uint32_t total_size;

  ~TensorBase() {
    if(data != nullptr) {
      free(data);
      DEBUG("TensorBase memory freed..\r\n");
    }
  }
};

template <class T>
class Tensor {
  std::shared_ptr<TensorBase<T>> s;  // short for states

  void init(vector<uint32_t>& v) {
    s = std::make_shared<TensorBase<T>>(TensorBase<T>());
    s->total_size = 0;

    for (auto i : v) {
      s->shape.push_back(i);
      // total_size = (total_size == 0)? i : total_size *= i;
      if (s->total_size == 0) {
        s->total_size = i;
      } else {
        s->total_size *= i;
      }
    }

    s->data = (T*)malloc(unit_size() * s->total_size);
    if(s->data == NULL) ERR_EXIT("ran out of memory for %lu malloc", unit_size() * s->total_size);
  }

 public:
  Tensor(void) {
    s = std::make_shared<TensorBase<T>>(TensorBase<T>());
    s->total_size = 0;
    s->data = nullptr;
  }

  Tensor(initializer_list<uint32_t> l) {
    vector<uint32_t> v;
    for (auto i : l) {
      v.push_back(i);
    }

    init(v);
  }

  Tensor(vector<uint32_t> v) { init(v); }

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
    s = nullptr;
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
}
#endif
