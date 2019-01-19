#ifndef UTENSOR_ARRAY_OPS
#define UTENSOR_ARRAY_OPS

#include "uTensor/util/uTensor_util.hpp"
#include "uTensor/util/quantization_utils.hpp"
#include "uTensor/core/uTensorBase.hpp"
#include <cstring>
#include <cmath>

//T = inferred
//mode = MIN_FIRST
//name = unspecified
template <typename T>
void QuantizeV2(S_TENSOR input, S_TENSOR _min_range, S_TENSOR _max_range,
                    S_TENSOR output, S_TENSOR output_min, S_TENSOR output_max) {

    float input_min_range = *(_min_range->read<float>(0, 0));
    float input_max_range = *(_max_range->read<float>(0, 0));

    if(input_max_range < input_min_range) ERR_EXIT("input_max_range must be larger than input_min_range.");

    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                   fabsf(input_max_range))) / 100.0f;
    TensorShape v;

    TensorShape org = input->getShape();
    for (size_t i = 0; i < org.size(); i++) {
        v.push_back(org[i]);
    }

    if(output && output->getSize() == 0) {
      output->resize(v);
    }

    float max_range = std::max(input_max_range, min_range + epsilon);
    max_range = std::max(0.0f, max_range);

    FloatToQuantizedStruct<T> f2q(min_range, max_range);

    //quantization_utils.h:149
    const float* input_ptr = input->read<float>(0, 0);
    T* output_ptr = output->write<T>(0, 0);
    float* output_min_ptr = output_min->write<float>(0, 0);
    float* output_max_ptr = output_max->write<float>(0, 0);

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

class QuantizeV2Op : public Operator {
  public:
    QuantizeV2Op() {
      n_inputs = 3;
      n_outputs = 3;
    }

    virtual void compute() override {
      QuantizeV2<unsigned char>(inputs[0], inputs[1], inputs[2],
              outputs[0], outputs[1], outputs[2]);
    }
}; 

//mode = MIN_FIRST
//name = unspecified
//dequantize_op.cc: 87
template <typename T>
void dequantize(S_TENSOR input, S_TENSOR min_range, S_TENSOR max_range, S_TENSOR output) {
    float min = *(min_range->read<float>(0, 0));
    float max = *(max_range->read<float>(0, 0));
      //auto tensor allocation
    TensorShape out_shape;
    output->resize(input->getShape());

    const T* input_ptr = input->read<T>(0, 0);
    float* output_ptr = output->write<float>(0, 0);

    //quantization_utils.h: 771
    QuantizedToFloatStruct<T> q2f(min, max);

    //quantization_utils.h: 141
    for(uint32_t i = 0; i < input->getSize(); i++) {
        float val = static_cast<float>(input_ptr[i]);
        output_ptr[i] = ((q2f.range_min_rounded - q2f.lowest_quantized() * q2f.range_scale) + \
                        val * q2f.range_scale);
    }
}
class DequantizeOp : public Operator {
  public:
    DequantizeOp() {
      n_inputs = 3;
      n_outputs = 1;
    }

    virtual void compute() override {
      dequantize<unsigned char>(inputs[0], inputs[1], inputs[2],
              outputs[0]);
    }
};
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


//Pre:
//output.getShape == shape, or
//output.getSize() == 0, in which case, a new tensor is allocated and assigned to the referenced output
//Post:
//input content copied into output with output.getShape == shape

///NT: This Op hasn't been tested extensively. We will have to increase the test-coverage for this function.
template <typename T>
void reshape(S_TENSOR input, S_TENSOR shape, S_TENSOR output) {
    TensorShape dim;

    auto shape_vec = shape->getShape();
    for(size_t i = 0; i < shape_vec.size(); i++) {
        //This may due to references to zero-tensor's dimensions
        if(shape_vec[i] == 0) ERR_EXIT("shape tensor contains 0 value entry");
    }


    //validating and inferring dimensions
    int infer_index = -1;
    uint32_t dim_rem = input->getSize();
    const int* val = shape->read<int>(0, 0);
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


    const T* input_ptr = input->read<T>(0, 0);
    //check if the output dim is valid
    if(output && output->getSize() > 0 && dim == output->getShape()) {
        //copy
        T* output_ptr = output->write<T>(0, 0);
        memcpy(output_ptr, input_ptr, (std::size_t) input->getSize_in_bytes());
    } else if(output && output->getSize() > 0 && dim != output->getShape()) {
        ERR_EXIT("output tensor dimension mismatches supplied shape")
    } else {
        //construct a new tensor and copy
        output->resize(dim);
        T* output_ptr = output->write<T>(0, 0);
        memcpy(output_ptr, input_ptr, (std::size_t) input->getSize_in_bytes());
    }

}

class ReshapeOp : public Operator {
  public:
    ReshapeOp() {
      n_inputs = 2;
      n_outputs = 1;
    }

    virtual void compute() override {
      reshape<float>(inputs[0], inputs[1], outputs[0]);
    }
};

/*
https://github.com/tensorflow/tensorflow/blob/f7ec99516/tensorflow/core/kernels/quantized_reshape_op.cc
https://github.com/tensorflow/tensorflow/blob/f7ec99516/tensorflow/core/kernels/reshape_op.h
*/
class QuantizedReshapeOp : public ReshapeOp {
  public:
    QuantizedReshapeOp() {
      n_inputs = 4;
      n_outputs = 3;
    }

    virtual void compute() override {
      reshape<uint8_t>(inputs[0], // input tensor
                       inputs[1], // shape
                       outputs[0]);
      S_TENSOR input_min = this->inputs[2];
      S_TENSOR input_max = this->inputs[3];
      S_TENSOR output_min = this->outputs[1];
      S_TENSOR output_max = this->outputs[2];
      *(output_min->write<float>(0, 0)) = *(input_min->read<float>(0, 0));
      *(output_max->write<float>(0, 0)) = *(input_max->read<float>(0, 0));
    }
};

#endif  //UTENSOR_ARRAY_OPS
