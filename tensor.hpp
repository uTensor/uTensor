#ifndef TENSOR_H
#define TENSOR_H

#include "mbed.h"
#include<memory>
#include <vector>
#include <initializer_list>
#include <stdlib.h>

//TODO:
//Make a Tensor class, a datum class
//With Vector to hold internal struct
//Use Variadic functions for APIs
//Transvers via the lowest dimension
//Template class for the data types

//using namespace std;

template <class T>
class TensorBase {
    private:
    vector<uint32_t> shape;
    std::shared_ptr<T> data;
    uint32_t total_size;
    uint16_t unit_size;

    public:
    TensorBase(void) {
        total_size = 0;
    }

    TensorBase(initializer_list<uint32_t> l) {
        total_size = 0;
        for(auto i:l) {
            shape.push_back(i);
            total_size = (total_size == 0)? i : total_size *= i;
        }
        
        unit_size = sizeof(T);
        data = std::shared_ptr<T> ((T*)malloc(unit_size * total_size), free);
        printf("total size is:%d\r\n", total_size);
    
    }

    TensorBase(vector<uint32_t> v) {
        total_size = 0;
        for(auto i:v) {
            shape.push_back(i);
            total_size = (total_size == 0)? i : total_size *= i;
        }

        unit_size = sizeof(T);
        data = std::shared_ptr<T> ((T*)malloc(unit_size * total_size), free);
        printf("total size is:%d\r\n", total_size);
    
    }

    //returns how far a given dimension is apart
    size_t getStride(size_t dim_index) {
        unsigned int size_accm = 1;
        for(auto it = shape.begin() + dim_index + 1; it != shape.end(); it++) {
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

        printf("p_offset: %d\r\n", p_offset);
        return data.get() + p_offset;
    }

    T* getPointer(vector<uint32_t> v) {
        size_t p_offset = 0;
        signed short current_dim = 0;
        for(auto i:v) {
            p_offset += i * getStride(current_dim);
            current_dim++;
        }

        printf("p_offset: %d\r\n", p_offset);

        return data.get() + p_offset;
    }

    vector<uint32_t>& getShape() {
        return shape;
    }

    uint32_t getSize() {
        return total_size;
    }

    uint32_t getSize_in_bytes() {
        return total_size * unit_size;
    }

    //returns the number of dimensions in the tensor
    size_t getDim() {
        return shape.size();
    }

    ~TensorBase() {
        // if(data != NULL)
        //     free(data);

        printf("fuckers\r\n");
    }
};


#endif