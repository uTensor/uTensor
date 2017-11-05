#ifndef UTENSOR_ARRAY_OPS
#define UTENSOR_ARRAY_OPS

#include <cstring>
#include <math.h>
#include "uTensor_util.hpp"
#include "quantization_utils.hpp"

//T = inferred
//mode = MIN_FIRST
//name = unspecified
template <typename T>
void QuantizeV2(Tensor* input, Tensor* _min_range, Tensor* _max_range,
                    Tensor* output, Tensor* output_min, Tensor* output_max) {

    float input_min_range = *(_min_range->read<float>(0, 0));
    float input_max_range = *(_max_range->read<float>(0, 0));

    if(input_max_range < input_min_range) ERR_EXIT("input_max_range must be larger than input_min_range.");

    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                   fabsf(input_max_range))) / 100.0f;

    float max_range = std::max(input_max_range, min_range + epsilon);
    max_range = std::max(0.0f, max_range);

    FloatToQuantizedStruct<T> f2q(min_range, max_range);

    //quantization_utils.h:149
    float* input_ptr = input->read<float>(0, 0);
    T* output_ptr = output->write<T>(0, 0);
    float* output_min_ptr = output_min->read<float>(0, 0);
    float* output_max_ptr = output_max->read<float>(0, 0);

    ///NT: need error checking at some point...
    for(uint32_t i = 0; i < input->getSize(); i++) {
        float val = std::round(input_ptr[i] * f2q.range_scale);
        val -= f2q.range_min_scaled - f2q.lowest_quantized();
        val = std::max(val, f2q.lower_bound_float());
        val = std::min(val, f2q.upper_bound_float());
        uint32_t intTmp = static_cast<uint32_t>(val); ///NT: omit this?
        output_ptr[i] = static_cast<T>(intTmp);
    }

    *output_min_ptr = min_range;
    *output_max_ptr = max_range;
    
}

//mode = MIN_FIRST
//name = unspecified
//dequantize_op.cc: 87
template <typename T>
void dequantize(Tensor* input, Tensor* min_range, Tensor* max_range, Tensor* output) {
    float min = *(min_range->read<float>(0, 0));
    float max = *(max_range->read<float>(0, 0));
      //auto tensor allocation
    Shape out_shape;
    tensorChkAlloc<float>(&output, input->getShape());

    T* input_ptr = input->read<T>(0, 0);
    float* output_ptr = output->write<float>(0, 0);

    //quantization_utils.h: 771
    QuantizedToFloatStruct<T> q2f(min, max);

    //quantization_utils.h: 141
    for(uint32_t i = 0; i < input->getSize(); i++) {
        float val = static_cast<float>(input_ptr[i]);
        output_ptr[i] = ((q2f.range_min_rounded - q2f.lowest_quantized() * q2f.range_scale) + \
                        val * q2f.range_scale);
    }
/*
  number_of_steps = 1 << (# of bits in T)
  range_adjust = number_of_steps / (number_of_steps - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = range / number_of_steps
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
*/

    // for(uint32_t i = 0; i < input.getSize(); i++) {
    //     output_ptr[i] = QuantizedToFloat(input_ptr[i], min, max);
    // }

}

//Pre:
//output.getShape == shape, or
//output.getSize() == 0, in which case, a new tensor is allocated and assigned to the referenced output
//Post:
//input content copied into output with output.getShape == shape

///NT: This Op hasn't been tested extensively. We will have to increase the test-coverage for this function.
template <typename T>
void reshape(Tensor* input, Tensor* shape, Tensor** output) {
    Shape dim;

    //validating and inferring dimensions
    int infer_index = -1;
    uint32_t dim_rem = input->getSize();
    int* val = shape->read<int>(0, 0);
    for(uint32_t i = 0; i < shape->getSize(); i++) {
        if(val[i] == -1) {
            if(infer_index == -1) {
                infer_index = i;
            } else {
                ERR_EXIT("shape can only contain one inference (-1) at a time");
            }
        } else {
            dim_rem /= val[i];
        }

        dim.push_back(static_cast<uint32_t>(val[i]));
    }

    if(infer_index != -1) {
        dim[infer_index] = dim_rem;
        dim_rem = 1; // dim_rem / dim_rem = 1
    }

    if(dim_rem != 1) ERR_EXIT("supplied shape does not match up to input");


    T* input_ptr = input->read<T>(0, 0);
    //check if the output dim is valid
    if(*output && (*output)->getSize() > 0 && dim == (*output)->getShape()) {
        //copy
        T* output_ptr = (*output)->read<T>(0, 0);
        std::memcpy(output_ptr, input_ptr, (std::size_t) input->getSize_in_bytes());
    } else if(*output && (*output)->getSize() > 0 && dim != (*output)->getShape()) {
        ERR_EXIT("output tensor dimension mismatches supplied shape")
    } else {
        //construct a new tensor and copy
        Tensor* tmp = new RamTensor<T>(dim);
        T* output_ptr = tmp->write<T>(0, 0);
        std::memcpy(output_ptr, input_ptr, (std::size_t) input->getSize_in_bytes());
        *output = tmp;
    }

}

#endif  //UTENSOR_ARRAY_OPS
