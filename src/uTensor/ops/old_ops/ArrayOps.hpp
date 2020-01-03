#ifndef UTENSOR_ARRAY_OPS
#define UTENSOR_ARRAY_OPS

#include "src/uTensor/util/uTensor_util.hpp"
#include "src/uTensor/util/quantization_utils.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include <cstring>
#include <cmath>
#include <array>

/*
 * Requires proper testing, tested only for some cases
 * Not supported features: Elipses_axis mask, new_axis_mask, undefined dimension size (dim == -1)
 */
template <typename T>
void StridedSlice1D(S_TENSOR input, S_TENSOR output, int dim_index, std::vector<int> indexes)
{
    TensorShape output_shape;
    for(int i  = 0; i < input->getDim(); i++)
    {
        if(i != dim_index)
        {
            output_shape.push_back(input->getShape().at(i));
        }
        else  output_shape.push_back(indexes.size());
    }

    int batch = 1;
    int step = 1;
    for(int i = dim_index; i < input->getDim(); i++)
    {
        step *= input->getShape().at(i);
    }
    for(int i = dim_index + 1; i < input->getDim(); i++)
    {
        batch *= input->getShape().at(i);
    }


    output->resize(output_shape);
    T* out_ptr = output->write<T>(0, 0);
    const T* in_ptr = input->read<T>(0, 0);
    for(int i = 0; i < output->getSize(); i++)
    {
        int begin = indexes[(i / batch) % indexes.size()] * batch;
        int dim_offset = (i / (batch * indexes.size())) * step;
        int index = (i % batch);

        int offset = dim_offset + begin + index;
        out_ptr[i] = in_ptr[offset];
    }
}

/*
 * Requires proper testing, tested only for some cases
 * Not supported features: Elipses_axis mask, new_axis_mask, undefined dimension size (dim == -1)
 */
template <typename T>
void StridedSlice(S_TENSOR input, S_TENSOR begin_tensor, S_TENSOR end_tensor, S_TENSOR strides_tensor, S_TENSOR output, int begin_mask, int ellipsis_mask,
                  int end_mask, int new_axis_mask, int shrink_axis_mask )
{

    if (begin_tensor->getDim() != 1 || end_tensor->getDim() != 1 || strides_tensor->getDim() != 1
     || strides_tensor->getSize() != begin_tensor->getSize() || strides_tensor->getSize() != end_tensor->getSize()
     || strides_tensor->getSize() != input->getDim() )
    {
        ERR_EXIT("StridedSlice: Expected begin, end, and strides to be 1D equal size tensors, which size is equal to input_tensor dimensionality (new mask && elipsis mask not supported). "
                 "Input dimensionality:\"%d\", begin_tensor size: \"%d\", end_tensor size: \"%d\" and strides_tensor size: \"%d\".", input->getDim(), begin_tensor->getSize(), end_tensor->getSize(), strides_tensor->getSize());
    }

    if(input->getDim() >= 32)
    {
        ERR_EXIT("StridedSlice: Only 32bit masks are supported, input vector cannot have higher dimensionality than 32")
    }

    //TODO
    if(ellipsis_mask)
    {
        ERR_EXIT("StridedSlice: Ellipsis mask not supported yet");
    }

    //TODO
    if(new_axis_mask)
    {
        ERR_EXIT("StridedSlice: New axis mask not supported yet")
    }


    output->resize(input->getShape());
    memcpy(output->write<void>(0, 0), input->read<void>(0,0), input->getSize() * sizeof (T));

    const int* begin_ptr = begin_tensor->read<int>(0, 0);
    const int* end_ptr = end_tensor->read<int>(0, 0);
    const int* strides_ptr = strides_tensor->read<int>(0, 0);
    for (int i = 0; i < input->getDim(); ++i)
    {
      int begin_i = begin_ptr[i];
      int end_i = end_ptr[i];
      int stride_i = strides_ptr[i];
      int32_t dim_i = input->getShape().at(i);

      if (stride_i == 0) {
          ERR_EXIT("StridedSlice: strides[\"%d\"] must be non-zero", i);
      }

      bool shrink_i = (shrink_axis_mask & (1 << i));
      if(shrink_i)
      {
          end_i = begin_i + 1;
          stride_i = 1;

      }

      bool begin_masked_i = (begin_mask & (1 << i));
      bool end_masked_i = (end_mask & (1 << i));
      const std::array<int64_t, 2> valid_range = {
          {stride_i > 0 ? 0 : -1, stride_i > 0 ? (int64_t)dim_i : (int64_t)(-dim_i - 1)}};

      if(begin_masked_i) begin_i = valid_range[0];
      if(end_masked_i) end_i = valid_range[1];

      int64_t interval_length = (end_i - begin_i);
      int64_t size_i = interval_length / stride_i +
              (interval_length % stride_i != 0 ? 1 : 0);

      if(size_i == dim_i && stride_i == 1) continue; //whole dimension is taken unchanged
      else
      {

          std::vector<int> indexes;
          //Reversing the dimension
          if(begin_i < 0 && end_i < 0 && stride_i < 0)
          {
              begin_i += dim_i;
              end_i += dim_i;
              for(int j = 0; j < size_i; j++)
              {
                  int index = begin_i + j * stride_i;
                  if(index <= end_i) index = end_i + 1;
                  indexes.push_back(index);
              }
          }
          else if(begin_i >= 0 && end_i > 0 && stride_i > 0)
          {
              for(int j = 0; j < size_i; j++)
              {
                  int index = begin_i + j * stride_i;
                  if(index >= end_i) index = end_i - 1;
                  indexes.push_back(index);
              }
          }
          else {
              ERR_EXIT("Slided Strice: begin, and and stride must be all less than zero for the dimension or none of them")
          }

          S_TENSOR dim_input(new RamTensor<T>(output->getShape()));
          memcpy(dim_input->write<T>(0, 0), output->read<T>(0, 0), output->getSize() * sizeof (T));
          StridedSlice1D<T>(dim_input, output, i, indexes);

      }
    }

    TensorShape shrinked_shape;
    for (int i = 0; i < output->getDim(); i++)
    {
        bool shrink_i = (shrink_axis_mask & (1 << i));
        if(!shrink_i)
        {
           shrinked_shape.push_back(output->getShape().at(i));
        }
    }
    if(shrinked_shape.size() == 0) shrinked_shape.push_back(1);
    //the shrinked dimension size was 1 so no data update needed
    output->resize(shrinked_shape);
}

template<typename T>
class StridedSliceOp : public Operator {
  public:
  StridedSliceOp(int begin_mask, int ellipsis_mask,
               int end_mask, int new_axis_mask,
               int shrink_axis_mask) : _begin_mask(begin_mask), _ellipsis_mask(ellipsis_mask), _end_mask(end_mask),
                                      _new_axis_mask(new_axis_mask), _shrink_axis_mask(shrink_axis_mask)  {
    n_inputs = 4;
    n_outputs = 1;
  }
  virtual void compute() override {
    StridedSlice<T>(inputs[0], inputs[1], inputs[2], inputs[3], outputs[0],
                         _begin_mask, _ellipsis_mask, _end_mask,
                         _new_axis_mask, _shrink_axis_mask);
  }

  protected:
  int _begin_mask, _ellipsis_mask, _end_mask;
  int _new_axis_mask, _shrink_axis_mask;
};



template <typename T>
void Pack(unsigned int N, S_TList inputs, S_TENSOR output, int axis )
{
    if (axis > 0)
    {
        ERR_EXIT("Pack: axis parametr other than 0 is not supported by uTensor");
    }
    if(N < 2)
    {
        ERR_EXIT("Pack operation requires at leas 2 input vectors");
    }
    if(N != inputs.size())
    {
        ERR_EXIT("Pack: Mistmatch between input parameter and inputs count");
    }


    std::vector<const T*> input_ptrs;
    size_t dim_1 = inputs[0]->getDim();
    size_t input_size = inputs[0]->getSize();
    TensorShape shape_1 = inputs[0]->getShape();
    for(unsigned int i = 0; i < N; i++)
    {
        if(inputs[i]->getDim() != dim_1)
        {
            ERR_EXIT("Pack: input vectors must have same dimensions");
        }
        for(unsigned int j = 0; j < dim_1; j++)
        {
            if(inputs[i]->getShape().at(j) != shape_1.at(j))
            {
                ERR_EXIT("Pack: input vectors must have identical shape");
            }
        }
        input_ptrs.push_back(inputs[i]->read<T>(0,0));
    }

    TensorShape output_shape;
    for (size_t i = 0; i < dim_1; i++) {
        output_shape.push_back(shape_1.at(i));
    }
    output_shape.push_back(N);

    if(output && output->getSize() == 0) {
      output->resize(output_shape);
    }

    T* output_ptr = output->write<T>(0,0);
    for(int i = 0; i < input_size; i++)
    {
        for(int j = 0; j < N; j++)
        {
            output_ptr[j * input_size + i] = input_ptrs[j][i];
        }
    }
}

template<typename T>
class PackOp : public Operator {
  public:
  PackOp(unsigned int N, int axis) : _NN(N),_axis(axis) {
    n_inputs = N;
    n_outputs = 1;
  }
  virtual void compute() override {

      Pack<T>(_NN, inputs, outputs[0], _axis);
  }

  protected:
  /*const*/ int _NN;
  int _axis;
};


//T = inferred
//mode = MIN_FIRST
//name = unspecified
template <typename T>
void Shape(S_TENSOR input, S_TENSOR output) {

    TensorShape input_shape = input->getShape();
    TensorShape output_shape;
    output_shape.push_back(input->getDim());

    if(output && output->getSize() == 0) {
      output->resize(output_shape);
    }

    T* output_ptr = output->write<T>(0, 0);
    for (uint32_t i = 0; i < input->getDim(); i++) {
        output_ptr[i] = static_cast<T>(input_shape[i]);
    }
}

class ShapeOp : public Operator {
  public:
    ShapeOp() {
      n_inputs = 1;
      n_outputs = 1;
    }

    virtual void compute() override {
      Shape<int>(inputs[0], outputs[0]);
    }
};

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
        float val = ::round(input_ptr[i] * f2q.range_scale);
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

template <typename T>
void gather(S_TENSOR input, S_TENSOR indices, S_TENSOR output) {
    const T* input_ptr = input->read<T>(0,0);
    if (!output->getSize())
        output->resize(indices->getShape());
    T* out_ptr = output->write<T>(0,0);
    const uint32_t* indices_ptr = indices->read<uint32_t>(0,0); //Can probably templatize this 

    for(uint32_t i = 0; i < indices->getSize(); i++){
        if(indices_ptr[i] > input->getSize())
            ERR_EXIT("Gather indices out of input bound");
        T t = input_ptr[indices_ptr[i]];
        out_ptr[i] = t;
    }

}

/**
 * Gather V2 expects an additional axis parameter, for now we ignore this
 */
template <typename T>
class GatherOp : public Operator {
  public:
    GatherOp() {
      n_inputs = 3; // TODO add axis support
      n_outputs = 1;
    }

    virtual void compute() override {
      gather<T>(inputs[0], inputs[1], outputs[0]);
    }
};

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
