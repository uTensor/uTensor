#ifndef UTENSOR_NN_OPS
#define UTENSOR_NN_OPS

#include "uTensor/util/quantization_utils.hpp"
#include "uTensor/core/tensor.hpp"

template <class TIn, class T2, class TOut>
void Relu(S_TENSOR input, S_TENSOR in_min, S_TENSOR in_max,
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
class ReluOp : public Operator {
  public:
  ReluOp() {
    n_inputs = 3;
    n_outputs = 3;
  }
  virtual void compute() override {
    Relu<T1, T2, TOut>(inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], outputs[2]);
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
                       Padding padding) {
  Shape in_shape = input->getShape();
  uint32_t n_batch = in_shape[0];
  uint32_t in_rows = in_shape[1];
  uint32_t in_cols = in_shape[2];
  uint32_t in_channels = in_shape[3];

  size_t half_window_rows = window_rows / 2;
  size_t half_window_cols = window_cols / 2;

  uint32_t out_rows, out_cols;
  size_t row_start_idx, col_start_idx;
  if (padding == VALID) {
    out_rows = (in_rows - window_rows) / row_stride + 1;
    out_cols = (in_cols - window_cols) / col_stride + 1;
    row_start_idx = half_window_rows;
    col_start_idx = half_window_cols;
  } else {
    out_rows = in_rows;
    out_cols = in_cols;
    row_start_idx = 0;
    col_start_idx = 0;
  }
  Shape out_shape;
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
      for (size_t c_row_idx = row_start_idx; c_row_idx + half_window_rows < in_rows; c_row_idx += row_stride) {
        for (size_t c_col_idx = col_start_idx; c_col_idx + half_window_cols < in_cols; c_col_idx += col_stride) {
          size_t total_offset = idx_batch*in_batch_stride + idx_chnl*in_chnl_stride +
                                c_row_idx*in_row_stride + c_col_idx*in_col_stride;
          T max_value = *(input->read<T>(total_offset, 0));
          // scanning window
          for (size_t i = -half_window_rows; i <= half_window_rows; ++i) {
            for (size_t j = -half_window_cols; j <= half_window_cols; ++j) {
              T current_value;
              if (c_row_idx + i < 0 ||
                  c_row_idx + i >= in_rows ||
                  c_col_idx + j < 0 ||
                  c_col_idx + j >= in_cols) {
                    // padding with zero
                    current_value = 0;
              } else {
                size_t offset = total_offset + i*in_row_stride + j*in_col_stride;
                current_value = *(input->read<T>(offset, 0));
              }
              if (current_value > max_value) {
                max_value = current_value;
              }
            }
          }
          // write output
          size_t out_row_idx = (c_row_idx - row_start_idx) / row_stride;
          size_t out_col_idx = (c_col_idx - col_start_idx) / col_stride;
          size_t out_offset = idx_batch*out_batch_stride + idx_chnl*out_chnl_stride +
                              out_row_idx*out_row_stride + out_col_idx*out_col_stride;
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
    SpatialMaxPooling<T>(this->inputs[0], this->outputs[0],
                         this->_window_rows, this->_window_cols,
                         this->_row_stride, this->_col_stride,
                         this->_padding);
    S_TENSOR in_min_tensor = this->inputs[1];
    S_TENSOR in_max_tensor = this->inputs[2];
    float in_min_value = *(in_min_tensor->read<float>(0, 0));
    float in_max_value = *(in_max_tensor->read<float>(0, 0));

    S_TENSOR out_min_tensor = this->outputs[1];
    S_TENSOR out_max_tensor = this->outputs[2];
    *(out_min_tensor->write<float>(0, 0)) = in_min_value;
    *(out_max_tensor->write<float>(0, 0)) = in_max_value;
  }
};

#endif  // UTENSOR_NN_OPS
