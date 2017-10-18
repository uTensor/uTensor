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

    void init(vector<uint32_t> &v) { s = std::make_shared<TensorBase<T>>(TensorBase<T>());
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

    vector<uint32_t>& getShape(void) {
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

//Usage:
//Tensor<int> inputTensor({10,10,100,40});
//vector<uint8_t> permute = {2,3,0,1};
//
//permuteIndexTransform trans(inputTensor.getShape(), permute);
//
//Tensor<int> outputTensor(trans.getNewShape());  //of shape {100,40,10,10}
//size_t output_buffer_index = trans[input_buffer_index];

class permuteIndexTransform {
private:
    vector<uint8_t> permute;
    vector<uint8_t> unpermute;
    Shape in_shape;
    Shape in_stride;
    Shape out_shape;
    Shape out_stride;

    void computeOutputShape(void) {
        out_stride.clear();
        if(in_shape.empty()) ERR_EXIT("input shape not set");
        if(permute.empty() || permute.size() != in_shape.size())
            ERR_EXIT("invalid permute vector");

        for(auto&& curr_axis:permute) {
            out_shape.push_back(in_shape[curr_axis]);
        }

    }

    size_t evalStride(size_t dim_index, Shape s) {
        unsigned int size_accm = 1;
        for(auto it = s.begin() + dim_index + 1; it != s.end(); it++) {
            size_accm *= *it;
        }

        return (size_t) size_accm;
    }

/*    void computeOutputStride(void) {
        out_stride.clear();
        for(uint32_t i = 0; i < out_shape.size(); i++) {
            out_stride.push_back(evalStride(i, out_shape));
        }
    }*/
    void computeInputStride(void) {
        in_stride.clear();
        for(uint32_t i = 0; i < in_shape.size(); i++) {
            in_stride.push_back(evalStride(i, in_shape));
        }
    }
    void computeOutputStride(void) {
        out_stride.clear();
        for(uint32_t i = 0; i < out_shape.size(); i++) {
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
    void setPermute(vector<uint8_t> &_permute) { 
        permute = _permute; 
        unpermute.resize(permute.size());
        uint8_t i = 0;
        for (auto a : permute) {
            unpermute[a] = i;
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

    // size_t forward(size_t index) {
    //     size_t out_index = 0;
    //     size_t rem = index;

    //     for(size_t curr_dim = 0; curr_dim < in_shape.size(); curr_dim++) {
    //         size_t curr_dim_size = in_shape[curr_dim];
    //         out_index += (rem / curr_dim_size) * out_stride[curr_dim];
    //         rem = rem % curr_dim_size;
    //     }

    //     return out_index;
    // }

    size_t operator[] (const size_t index)
    {
        size_t out_index = 0;
        size_t rem = index;

        for(size_t curr_dim = 0; curr_dim < in_shape.size(); curr_dim++) {
            size_t curr_stride = in_stride[curr_dim];
            out_index += (rem / curr_stride) * out_stride[unpermute[curr_dim]];
            rem = rem % curr_stride;
        }

        out_index += rem;

        return out_index;
    }
     /*size_t operator[] (const size_t index)
            {
                size_t out_index = 0;
                size_t rem = index;

                for(size_t curr_dim = 0; curr_dim < in_shape.size(); curr_dim++) {
                    size_t curr_dim_size = in_shape[curr_dim];

                    out_index += (rem / curr_dim_size) * out_stride[curr_dim];
                    rem = rem % curr_dim_size;
                }

                out_index += rem;

                return out_index;
            } */
};

#endif
