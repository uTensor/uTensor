#ifndef UTENSOR_TENSOR_H
#define UTENSOR_TENSOR_H

#include "mbed.h"
#include <memory>
#include <vector>
#include <initializer_list>
#include "stdlib.h"
#include <uTensor_util.hpp>

template <class U>
class TensorBase {
public:
    vector<uint32_t> shape;
    U* data;
    uint32_t total_size;

    ~TensorBase() {
        DEBUG("TensorBase destruction..\r\n");
        free(data);
    }
};

template <class T>
class Tensor {
    std::shared_ptr<TensorBase<T>> s; //short for states

    void init(vector<uint32_t> &v) {

        s = std::make_shared<TensorBase<T>>(TensorBase<T>());
        s->total_size = 0;
        
        for(auto i:v) {
            s->shape.push_back(i);
            //total_size = (total_size == 0)? i : total_size *= i;
            if(s->total_size == 0) {
                s->total_size = i;
            } else {
                s->total_size *= i;
            }
        }

        
        s->data = (T*) malloc(unit_size() * s->total_size);
        //printf("total size is:%d\r\n", (int) total_size);
    }


    public:

    Tensor(void) {
        s->total_size = 0;
    }

    Tensor(initializer_list<uint32_t> l) {
        vector<uint32_t> v;
        for(auto i:l) {
            v.push_back(i);
        }

        init(v);
    }

    Tensor(vector<uint32_t> v) {
        init(v);
    }

    //returns how far a given dimension is apart
    size_t getStride(size_t dim_index) {
        unsigned int size_accm = 1;
        for(auto it = s->shape.begin() + dim_index + 1; it != s->shape.end(); it++) {
            size_accm *= *it;
        }

        return (size_t) size_accm;
    }

    //PRE:      l, initization list, specifying the element/dimension
    //POST:     When a degenerative index is supplied, the pointer
    //          lowest specified dimension is returned.
    //          Otherwise, return the pointer to the specific element.
    T* getPointer(initializer_list<size_t> l) {
        size_t p_offset = 0;
        signed short current_dim = 0;
        for(auto i:l) {
            p_offset += i * getStride(current_dim);
            current_dim++;
        }

        //printf("p_offset: %d\r\n", p_offset);
        return s->data + p_offset;
    }

    T* getPointer(vector<uint32_t> v) {
        size_t p_offset = 0;
        signed short current_dim = 0;
        for(auto i:v) {
            p_offset += i * getStride(current_dim);
            current_dim++;
        }

        printf("p_offset: %d\r\n", p_offset);

        return s->data + p_offset;
    }

    vector<uint32_t> getShape(void) {
        return s->shape;
    }

    uint32_t getSize(void) {
        return s->total_size;
    }

    uint16_t unit_size(void) {
        return sizeof(T);
    }

    uint32_t getSize_in_bytes(void) {
        return s->total_size * unit_size();
    }

    //returns the number of dimensions in the tensor
    size_t getDim(void) {
        return s->shape.size();
    }

    ~Tensor() {
        // if(data != NULL)
        //     free(data);

        DEBUG("Tensor Destructed\r\n");
    }
};

template<typename Tin, typename Tout>
Tensor<Tout> TensorCast(Tensor<Tin> input) {
  Tensor<Tout> output(input.getShape());
  Tin* inputPrt = input.getPointer({});
  Tout* outputPrt = output.getPointer({});
  
  for(uint32_t i = 0; i < input.getSize(); i++) {
    outputPrt[i] = static_cast<Tout>(inputPrt[i]);
  }

  return output;
}

template<typename T>
Tensor<T> TensorConstant(vector<uint32_t> shape, T c) {
  Tensor<T> output(shape);
  T *outPrt = output.getPointer({});
  
  for(uint32_t i = 0; i < output.getSize(); i++) {
    outPrt[i] = c;
  }

  return output;
}

template<typename T>
Tensor<T> TensorConstant(initializer_list<uint32_t> l, T c) {
    vector<uint32_t> v;
    for(auto i:l) {
        v.push_back(i);
    }

    return TensorConstant<T>(v, c);
}


#endif
