#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include "src/uTensor/util/quantization_utils.hpp"
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
#include <math.h>
#include <algorithm>


template <class Tin, class Tout>
void Softmax(S_TENSOR input, S_TENSOR output)
{
    size_t dim = input->getDim(), row_dim_max, col_dim_max, row_stride, col_stride;
    if (dim > 2 || dim < 1) {
        ERR_EXIT("Softmax only supports 1D or 2D tensor");
    }
    if (dim == 1) {
        row_dim_max = 1;
        col_dim_max = input->getShape().at(0);
        row_stride = 0;
        col_stride = input->getStride(0);
    } else {
        row_dim_max = input->getShape().at(0);
        col_dim_max = input->getShape().at(1);
        row_stride = input->getStride(0);
        col_stride = input->getStride(1);
    }

    if (output && output->getSize() == 0) {
        output->resize(input->getShape());
    }
    Tout* out_ptr = output->write<Tout>(0,0);
    const Tin* in_ptr = input->read<Tin>(0, 0);

    for (size_t row_dim = 0; row_dim < row_dim_max; ++row_dim) {
        size_t base_offset = row_dim * row_stride;

        Tout max = 0;

        // calculate the max first, as we need to subtract max from every value
        // this avoids Infinity results when calling exp() later
        for (size_t col_dim = 0; col_dim < col_dim_max; ++col_dim) {
            size_t offset = base_offset + col_dim * col_stride;
            Tout logit = (Tout) in_ptr[offset];
            if (logit > max) {
                max = logit;
            }
        }

        Tout reduce_sum = 0;
        for (size_t col_dim = 0; col_dim < col_dim_max; ++col_dim) {
            size_t offset = base_offset + col_dim * col_stride;
            Tout logit = (Tout) in_ptr[offset];
            logit -= max;
            reduce_sum += exp(logit);
        }
        for (size_t col_dim = 0; col_dim < col_dim_max; ++col_dim){
            size_t offset = base_offset + col_dim * col_stride;
            Tout logit = (Tout) in_ptr[offset];
            logit -= max;
            out_ptr[offset] = exp(logit) / reduce_sum;
        }
    }
}

template <class Tin, class Tout>
class SoftmaxOp : public Operator {
  public:
  SoftmaxOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
    Softmax<Tin, Tout>(inputs[0], outputs[0]);
  }
};

template <class TIn, class TOut>
void Relu(S_TENSOR input,
          S_TENSOR output) {

  const TIn* in = input->read<TIn>(0, 0);
  if (output && output->getSize() == 0) {
      output->resize(input->getShape());
  }
  TOut* out = output->write<TOut>(0, 0);
  for (uint32_t i = 0; i < output->getSize(); i++) {
    if (in[i] > 0.0) {
      out[i] = in[i];
    } else {
      out[i] = 0.0;
    }
  }
}

template<class T1, class TOut>
class ReluOp : public Operator {
  public:
  ReluOp() {
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
    Relu<T1, TOut>(inputs[0], outputs[0]);
  }
};


template <class TIn, class T2, class TOut>
void QuantizedRelu(S_TENSOR input, S_TENSOR in_min, S_TENSOR in_max,
          S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {
  const float input_min = in_min->read<T2>(0, 0)[0];
  const float input_max = in_max->read<T2>(0, 0)[0];
  const TIn* in = input->read<TIn>(0, 0);

  const TOut min_as_quantized =
      FloatToQuantized<TOut>(0.0f, input_min, input_max);
  if (output && output->getSize() == 0) {
      output->resize(input->getShape());
  }
  TOut* out = output->write<TOut>(0, 0);
  for (uint32_t i = 0; i < output->getSize(); i++) {
    if (in[i] > min_as_quantized) {
      out[i] = in[i];
    } else {
      out[i] = min_as_quantized;
    }
  }
  T2* v_out_min = out_min->write<T2>(0, 0);
  *v_out_min = input_min;
  T2* v_out_max = out_max->write<T2>(0, 0);
  *v_out_max = input_max;
}

template<class T1, class T2, class TOut>
class QuantizedReluOp : public Operator {
  public:
  QuantizedReluOp() {
    n_inputs = 3;
    n_outputs = 3;
  }
  virtual void compute() override {
    QuantizedRelu<T1, T2, TOut>(inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2]);
  }
};

/**
 * https://github.com/tensorflow/tensorflow/blob/982549ea3423df4270ff154e5c764beb43d472da/tensorflow/core/kernels/pooling_ops_common.h
 * https://github.com/tensorflow/tensorflow/blob/40eef4473bda90442bb55fcc67842f097c024580/tensorflow/core/kernels/maxpooling_op.h
 * https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/core/kernels/quantized_pooling_ops.cc#L109
 * https://github.com/tensorflow/tensorflow/blob/982549ea3423df4270ff154e5c764beb43d472da/tensorflow/core/kernels/eigen_pooling.h#L64
 */
template<typename T>
void SpatialMaxPooling(S_TENSOR input, S_TENSOR output,
                       int window_rows, int window_cols,
                       int row_stride, int col_stride,
                       Padding padding, T pad_value = 0) {
  /*
  * Arguments
  * ---------
  * input : S_TENSOR
  *     the intput tensor, assuming format of `NHWC`
  * output : S_TENSOR
  *     the output tensor
  *
  * Notes
  * -----
  * - padding
  *   - https://www.tensorflow.org/api_guides/python/nn#convolution
  *   - https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
  */
  TensorShape in_shape = input->getShape();
  uint32_t n_batch = in_shape[0];
  uint32_t in_rows = in_shape[1];
  uint32_t in_cols = in_shape[2];
  uint32_t in_channels = in_shape[3];

  size_t out_rows, out_cols;
  int pad_top, pad_left;
  if (padding == VALID) {
    out_rows = ((size_t) ceil(((float)(in_rows - window_rows) + 1) / ((float)row_stride)));
    out_cols = ((size_t) ceil(((float)(in_cols - window_cols) + 1) / ((float)col_stride)));
    // no padding for VALID
    pad_top = 0;
    pad_left = 0;
  } else {
    // SAME padding
    out_rows = ((size_t) ceil(((float)in_rows) / ((float) row_stride)));
    out_cols = ((size_t) ceil(((float)in_cols) / ((float) col_stride)));
    if (in_rows % row_stride == 0) {
      pad_top = std::max(window_rows - row_stride, 0) / 2;
    } else {
      pad_top = std::max(window_rows - (((int) in_rows) % row_stride), 0) / 2;
    }
    if (in_cols % col_stride == 0) {
      pad_left = std::max(window_cols - col_stride, 0) / 2;
    } else {
      pad_left = std::max(window_cols - (((int) in_cols) % col_stride), 0) / 2;
    }
  }
  TensorShape out_shape;
  out_shape.clear();
  out_shape.push_back(n_batch);
  out_shape.push_back(out_rows);
  out_shape.push_back(out_cols);
  out_shape.push_back(in_channels);
  output->resize(out_shape);

  // strides
  size_t in_batch_stride = input->getStride(0);
  size_t in_row_stride = input->getStride(1);
  size_t in_col_stride = input->getStride(2);
  size_t in_chnl_stride = input->getStride(3);

  size_t out_batch_stride = output->getStride(0);
  size_t out_row_stride = output->getStride(1);
  size_t out_col_stride = output->getStride(2);
  size_t out_chnl_stride = output->getStride(3);

  for (size_t idx_batch = 0; idx_batch < n_batch; ++idx_batch) {
    for (size_t idx_chnl = 0; idx_chnl < in_channels; ++idx_chnl) {
      size_t in_base_offset = idx_batch * in_batch_stride + idx_chnl * in_chnl_stride;
      for (int out_row_idx = 0; out_row_idx < out_rows; ++out_row_idx) {
        for (int out_col_idx = 0; out_col_idx < out_cols; ++out_col_idx) {
          T max_value;
          int base_row_idx = out_row_idx * row_stride - pad_top;
          int base_col_idx = out_col_idx * col_stride - pad_left;
          // if out of boundary, pad with pad_value
          if (base_row_idx < 0 ||
              base_row_idx >= in_rows ||
              base_col_idx < 0 ||
              base_col_idx >= in_cols) {
            max_value = pad_value;
          } else {
            size_t offset = in_base_offset +
                            ((size_t) base_row_idx) * in_row_stride +
                            ((size_t) base_col_idx) * in_col_stride;
            max_value = *(input->read<T>(offset, 0));
          }
          // scanning window
          for (int i = 0; i < window_rows; ++i) {
            for (int j = 0; j < window_cols; ++j) {
              T current_value;
              if (base_row_idx + i < 0 ||
                  base_row_idx + i >= in_rows ||
                  base_col_idx + j < 0 ||
                  base_col_idx + j >= in_cols) {
                current_value = pad_value;
              } else {
                size_t offset = in_base_offset +
                                ((size_t) base_row_idx + i) * in_row_stride +
                                ((size_t) base_col_idx + j) * in_col_stride;
                current_value = *(input->read<T>(offset, 0));
              }
              if (current_value > max_value) {
                max_value = current_value;
              }
            }
          }
          // write output
          size_t out_offset = idx_batch * out_batch_stride +
                              idx_chnl * out_chnl_stride +
                              out_row_idx * out_row_stride +
                              out_col_idx * out_col_stride;
          *(output->write<T>(out_offset, 0)) = max_value;
        }
      }
    }
  }
}

template<typename T>
class MaxPoolingOp : public Operator {
  public:
  MaxPoolingOp(int window_rows, int window_cols,
               int row_stride, int col_stride,
               Padding padding) : _window_rows(window_rows), _window_cols(window_cols),
                                  _row_stride(row_stride), _col_stride(col_stride) {
    _padding = padding;
    n_inputs = 1;
    n_outputs = 1;
  }
  virtual void compute() override {
    SpatialMaxPooling<T>(inputs[0], outputs[0],
                         _window_rows, _window_cols,
                         _row_stride, _col_stride, _padding);
  }

  protected:
  int _window_rows, _window_cols;
  int _row_stride, _col_stride;
  Padding _padding;
};

template<typename T>
class QuantizedMaxPoolingOp : public MaxPoolingOp<T> {
  public:
  QuantizedMaxPoolingOp(int window_rows, int window_cols,
                        int row_stride, int col_stride,
                        Padding padding) : MaxPoolingOp<T>(window_rows, window_cols, row_stride, col_stride, padding){
    this->n_inputs = 3;
    this->n_outputs = 3;
  }
  virtual void compute() override {
    S_TENSOR in_min_tensor = this->inputs[1];
    S_TENSOR in_max_tensor = this->inputs[2];
    float in_min = *(in_min_tensor->read<float>(0, 0));
    float in_max = *(in_max_tensor->read<float>(0, 0));

    // new range
    float new_in_max = in_max > 0 ? in_max : 0;
    float new_in_min = in_min < 0 ? in_min : 0;
    RequantizeManyInNewRange<T, T>(this->inputs[0].get(), this->inputs[0]->getSize(),
                                   in_min, in_max, new_in_min, new_in_max,
                                   this->inputs[0].get());
    // write new range
    S_TENSOR out_min_tensor = this->outputs[1];
    S_TENSOR out_max_tensor = this->outputs[2];
    *(out_min_tensor->write<float>(0, 0)) = new_in_min;
    *(out_max_tensor->write<float>(0, 0)) = new_in_max;
    // pooling
    T pad_value = FloatToQuantized<T>(0, new_in_min, new_in_max);
    SpatialMaxPooling<T>(this->inputs[0], this->outputs[0],
                         this->_window_rows, this->_window_cols,
                         this->_row_stride, this->_col_stride,
                         this->_padding,
                         pad_value);
  }
};

#endif  // UTENSOR_NN_OPS
