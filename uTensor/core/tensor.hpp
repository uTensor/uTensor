#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include "stdlib.h"
#include "uTensor_util.hpp"
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
 virtual std::string getName() { return name; }
 virtual void setName(std::string _name) {
    if(name == "") {
      name = _name;
    } else {
      ERR_EXIT("Tensor %s already has a name %s\r\n", _name.c_str(), name.c_str());
    }
  }


 virtual ~uTensor() = 0;
private:
 std::string name;

};

inline uTensor::~uTensor() {}
class TensorBase {
 public:
  std::vector<uint32_t> shape;
  void* data;
  uint32_t total_size;
  uint32_t cache_size;

  void initialize(std::vector<uint32_t>& vec) {
    uint32_t ret = 0;
    shape.clear();
    for (auto ele : vec) {
      shape.push_back(ele);
      if (ret == 0) {
          ret = ele;
      } else {
          ret *= ele;
      }
    }
    total_size = ret;
  }
  void allocate(uint8_t unit_size) {
    if (total_size > cache_size) {
      data = (void*)malloc(unit_size * cache_size);
    } else {
      data = (void*)malloc(unit_size * total_size);
    }
    if (data == NULL)
      ERR_EXIT("ran out of memory for %u malloc", unit_size * total_size);
  }

  ~TensorBase() {
    if (data != nullptr) {
      free(data);
      DEBUG("TensorBase memory freed..\r\n");
    }
  }
};

class Tensor : public uTensor {
  virtual void* read(size_t offset, size_t ele) { return nullptr; }
  virtual void* write(size_t offset, size_t ele) { return nullptr; }
  Tensor(const Tensor&);
  Tensor& operator=(const Tensor&);

 protected:
  std::shared_ptr<TensorBase> s;  // short for states
 public:
  Tensor() {
    s = std::make_shared<TensorBase>();
    s->total_size = 0;
    s->cache_size = std::numeric_limits<uint32_t>::max();
    s->data = nullptr;
    setName("");
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

  virtual void init(std::vector<uint32_t>& v) {

    s->initialize(v);
    if (s->data != NULL) {
        return;
    }
    s->allocate(unit_size());
  }

  virtual void resize(std::vector<uint32_t> v) {
      uint32_t size = s->total_size;

      s->initialize(v);

      if (size == s->total_size) {
          return;
      }

      s->allocate(unit_size());
  }

  std::vector<uint32_t> getShape(void) { return s->shape; }

  uint32_t getSize(void) { return s->total_size; }

  virtual uint16_t unit_size(void) { return 0; }

  uint32_t getSize_in_bytes(void) { return s->total_size * unit_size(); }

  // returns the number of dimensions in the tensor
  size_t getDim(void) { return s->shape.size(); }

  template <class T>
  const T* read(size_t offset, size_t ele) {
    return (const T*)read(offset, ele);
  }

  template <class T>
  T* write(size_t offset, size_t ele) {
    return (T*)write(offset, ele);
  }

  ~Tensor() {
    s = nullptr;
    DEBUG("Tensor Destructed\r\n");
  }
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

  size_t evalStride(size_t dim_index, Shape s) {
    unsigned int size_accm = 1;
    for (auto it = s.begin() + dim_index + 1; it != s.end(); it++) {
      size_accm *= *it;
    }

    return (size_t)size_accm;
  }

  void computeSStride(void) {
    s_stride.clear();
    for (uint32_t i = 0; i < s_shape.size(); i++) {
      s_stride.push_back(evalStride(i, s_shape));
    }
  }
  void computeLStride(void) {
    l_stride.clear();
    for (uint32_t i = 0; i < l_shape.size(); i++) {
      l_stride.push_back(evalStride(i, l_shape));
    }
  }

  void sortShape(Shape a, Shape b) {
    if(a.size() > b.size()) {
      l_shape = a;
      s_shape = b;
    } else if(a.size() < b.size()) {
      l_shape = b;
      s_shape = a;
      swap_flag = true;
    } else {
      auto it = std::find(a.begin(), a.end(), 1);
      if (it == a.end()) {
        l_shape = a;
        s_shape = b;
      } else {
        l_shape = b;
        s_shape = a;
        swap_flag = true;
      }
    }
  }

  void checkShape(void) {
    if(l_shape.size() < s_shape.size()) ERR_EXIT("cannot boardcast to fewer dimensions");
    for(int i = 0; i < (int) l_shape.size(); i++) {
      int small_i = i - (l_shape.size() - s_shape.size());
      if(small_i < 0) continue;
      if(l_shape[i] != s_shape[small_i] && s_shape[small_i] != 1) ERR_EXIT("ValueError: frames are not aligned");
      if(l_shape[i] < s_shape[small_i]) ERR_EXIT("Only single target broadcast is supported");
    }
  }


//b_shape being a smaller shape
 public:
  broadcastIndexTransform(Shape _l_shape, Shape _s_shape) {
    swap_flag = false;
    sortShape(_l_shape, _s_shape);
    checkShape();
    apply();
  }

  void apply(void) {
    //computeOutputShape();
    computeLStride();
    computeSStride();
  }

  Shape getOutputShape(void) {
    return l_shape;
  }

  bool is_swaped(void) {
    return swap_flag;
  }

  size_t operator[](const size_t linear_index) {
    size_t out_index = 0;
    size_t rem = linear_index;
    const size_t d_dim = l_shape.size() - s_shape.size();
    size_t s_dim;

    for (size_t curr_dim = 0; curr_dim < l_shape.size(); curr_dim++) {
      size_t curr_stride = l_stride[curr_dim];

      if(l_shape.size() - curr_dim <= s_shape.size()) {
        size_t curr_l_index = (rem / curr_stride);
        s_dim = curr_dim - d_dim;
        size_t curr_s_index = (curr_l_index % s_shape[s_dim]);
        out_index += curr_s_index * s_stride[s_dim];
      }

      rem = rem % curr_stride;
    }

    out_index += (rem % s_stride[s_dim]);

    // ///NT: DEBUG CODE
    // int sum = 0;
    // for(auto i:l_shape) {
    //   sum += i;
    // }
    // if((int)out_index > sum) {
    //   ERR_EXIT("index out of range, linear_index: %d, sum: %d, out_index: %d", linear_index, sum, out_index);
    // }
    // ///

    return out_index;
  }
};

#endif
