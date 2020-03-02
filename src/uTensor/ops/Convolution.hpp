#ifndef UTENSOR_CONVOLUTION_OPS_H
#define UTENSOR_CONVOLUTION_OPS_H
#include "operatorBase.hpp"
#include <algorithm>
#include <limits>

namespace uTensor {

enum Padding : uint8_t { VALID = 1, SAME = 2 };

// Can use these intermediate types to make the convolution operator more generic. Maxpool, conv, average pool, median etc. are all basically the same operation with target functions.
template <typename T>
class ConvFilter {
  T tmp;
  const Tensor& filter;

  public:
    ConvFilter(const Tensor& filter) : filter(filter) {}
    inline void reset() { tmp = 0; }
    inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
      const T filter_value =
        filter(i, j, k, l);
      tmp += (input_value * filter_value);

    }
    inline T finalize() const { return tmp; }
    inline const int16_t height() const { return filter->get_shape()[0]; }
    inline const int16_t width() const { return filter->get_shape()[1]; }
    inline const int16_t in_channels() const { return filter->get_shape()[2]; }
    inline const int16_t out_channels() const { return filter->get_shape()[3]; }
};

template <typename T>
class MaxFilter {
  T tmp;
  int16_t h;
  int16_t w;
  int16_t c;

  public: 
    MaxFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
    inline void reset() { tmp = std::numeric_limits<T>::lowest(); }
    inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
      tmp = std::max(tmp, input_value);
    }
    inline T finalize() const { return tmp; }
    inline const int16_t height() const { return h; }
    inline const int16_t width() const { return w; }
    inline const int16_t in_channels() const { return 1; }
    inline const int16_t out_channels() const { return c; }
};

template <typename T>
class AvgFilter {
  T tmp;
  int16_t w;
  int16_t h;
  int16_t c;

  public: 
    AvgFilter(int16_t w, int16_t h, int16_t c) : w(w), h(h), c(c) {}
    inline void reset() { tmp = 0; }
    inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
      tmp += input_value;
    }
    inline T finalize() const { return tmp / (static_cast<T>(w*h)); }
    inline const int16_t height() const { return h; }
    inline const int16_t width() const { return w; }
    inline const int16_t in_channels() const { return 1; }
    inline const int16_t out_channels() const { return c; }
};

template <typename T, typename Filter>
void generic_convolution_kernel(Tensor& out, const Tensor& in, Filter filter,
                        const Padding padding, const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = filter.out_channels();
  const int16_t filter_rows = filter.height();
  const int16_t filter_cols = filter.width();
  const int16_t filter_count = filter.out_channels();

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Each filter kernel produces one output channel.
        for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          //T output_val = 0;
          filter.reset();
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                T input_value;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  input_value = in(batch, in_y, in_x, in_channel);
                } else {
                  input_value = 0;
                }
                // size_t filter_index = filter_y * filter_cols * input_depth *
                // filter_count +
                //    filter_x * input_depth * filter_count +
                //    in_channel * filter_count + out_channel;
                // const T filter_value = filter(filter_index);
                filter.PartialCompute(input_value, filter_y, filter_x, in_channel, out_channel);
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = filter.finalize();
        }
      }
    }
  }
}
// Hack until I get the generic version working
template <typename T, typename Filter>
void generic_pool_convolution_kernel(Tensor& out, const Tensor& in, Filter filter,
                        const Padding padding, const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = input_depth; //filter.out_channels();
  const int16_t filter_rows = filter.height();
  const int16_t filter_cols = filter.width();
  //const int16_t filter_count = filter.out_channels();

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    //filter_left_offset =
    //    ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    //filter_top_offset =
    //    ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
    filter_left_offset = 
      (((input_cols - filter_cols)/stride_cols)*stride_cols + filter_cols - input_cols + 1)/2;
    filter_top_offset = 
      (((input_rows - filter_rows)/stride_rows)*stride_rows + filter_rows - input_rows + 1)/2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    // Each filter kernel produces one output channel.
    for (int out_channel = 0; out_channel < out_depth; ++out_channel) { // Thank god for no caches
      for (int out_y = 0; out_y < out_rows; ++out_y) {
        for (int out_x = 0; out_x < out_cols; ++out_x) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          //T output_val = 0;
          filter.reset();
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              //for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                T input_value;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  input_value = in(batch, in_y, in_x, out_channel);
                } else {
                  input_value = 0;
                }
                // size_t filter_index = filter_y * filter_cols * input_depth *
                // filter_count +
                //    filter_x * input_depth * filter_count +
                //    in_channel * filter_count + out_channel;
                // const T filter_value = filter(filter_index);
                filter.PartialCompute(input_value, filter_y, filter_x, out_channel, out_channel);
              //}
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = filter.finalize();
        }
      }
    }
  }
}

template <typename T>
void convolution_kernel(Tensor& out, const Tensor& in, const Tensor& filter,
                        const Padding padding, const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& f_shape = filter->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = f_shape[3];
  const int16_t filter_rows = f_shape[0];
  const int16_t filter_cols = f_shape[1];
  const int16_t filter_count = f_shape[3];

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Each filter kernel produces one output channel.
        for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          T output_val = 0;
          for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                T input_value;
                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                    (in_y < input_rows)) {
                  // Commenting out since these indices might be useful later
                  /*
                    size_t input_index = batch * input_rows * input_cols *
                    input_depth + in_y * input_cols * input_depth + in_x *
                    input_depth + in_channel; input_value =
                    in((uint32_t)input_index);
                   */
                  input_value = in(batch, in_y, in_x, in_channel);
                } else {
                  input_value = 0;
                }
                // size_t filter_index = filter_y * filter_cols * input_depth *
                // filter_count +
                //    filter_x * input_depth * filter_count +
                //    in_channel * filter_count + out_channel;
                // const T filter_value = filter(filter_index);
                const T filter_value =
                    filter(filter_y, filter_x, in_channel, out_channel);
                output_val += (input_value * filter_value);
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = output_val;
        }
      }
    }
  }
}

template <typename T>
class ConvOperator : public OperatorInterface<2, 1> {
 public:
  enum names_in : uint8_t { in, filter };
  enum names_out : uint8_t { out };
  ConvOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    ConvFilter<T> conv(*inputs[filter].tensor);
    generic_convolution_kernel<T, ConvFilter<T>>(*outputs[out].tensor, *inputs[in].tensor,
                          conv, _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
void depthwise_separable_convolution_kernel(Tensor& out, const Tensor& in,
                                            const Tensor& dw_filter,
                                            const Tensor& pw_filter,
                                            const Padding padding,
                                            const uint16_t (&strides)[4]) {
  const TensorShape& in_shape = in->get_shape();
  const TensorShape& df_shape = dw_filter->get_shape();
  const TensorShape& pf_shape = pw_filter->get_shape();

  const int16_t input_depth = in_shape[3];
  const int16_t input_rows = in_shape[1];
  const int16_t input_cols = in_shape[2];
  const int16_t input_batches = in_shape[0];
  const int16_t out_depth = pf_shape[3];
  const int16_t dw_filter_rows = df_shape[0];
  const int16_t dw_filter_cols = df_shape[1];
  const int16_t dw_filter_in_channels = df_shape[2];
  const int16_t dw_filter_channel_mult = df_shape[3];
  const int16_t pw_filter_in_channels = pf_shape[2];

  const int16_t stride_rows = strides[1];
  const int16_t stride_cols = strides[2];

  // Compute for now, but should assume codegen does this
  int16_t out_rows = out->get_shape()[1];
  int16_t out_cols = out->get_shape()[2];
  if (padding == VALID) {
    // out_rows = (input_rows - filter_rows) / stride_rows + 1;
    // out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    // out_rows = input_rows;
    // out_cols = input_cols;
  }
  // When we're converting the 32 bit accumulator to a lower bit depth, we
  int filter_left_offset;
  int filter_top_offset;
  if (padding == VALID) {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + dw_filter_cols - input_cols + 1) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + dw_filter_rows - input_rows + 1) / 2;
  } else {
    filter_left_offset =
        ((out_cols - 1) * stride_cols + dw_filter_cols - input_cols) / 2;
    filter_top_offset =
        ((out_rows - 1) * stride_rows + dw_filter_rows - input_rows) / 2;
  }

  // If we've got multiple images in our input, work through each of them.
  for (int batch = 0; batch < input_batches; ++batch) {
    // Walk through all the output image values, sliding the filter to
    // different positions in the input.
    for (int out_y = 0; out_y < out_rows; ++out_y) {
      for (int out_x = 0; out_x < out_cols; ++out_x) {
        // Fuse the Depthwise filtering with the pointwise filtering by iterating over the channels first
        for (int out_channel = 0; out_channel < out_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
          const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
          T output_val = 0;

          for (int filter_y = 0; filter_y < dw_filter_rows; ++filter_y) {
            for (int filter_x = 0; filter_x < dw_filter_cols; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                for (int r = 0; r < dw_filter_channel_mult; ++r) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  T input_value;
                  if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                      (in_y < input_rows)) {
                    // Commenting out since these indices might be useful later
                    /*
                      size_t input_index = batch * input_rows * input_cols *
                      input_depth + in_y * input_cols * input_depth + in_x *
                      input_depth + in_channel; input_value =
                      in((uint32_t)input_index);
                     */
                    input_value = in(batch, in_y, in_x, in_channel);
                  } else {
                    input_value = 0;
                  }
                  // size_t filter_index = filter_y * filter_cols * input_depth *
                  // filter_count +
                  //    filter_x * input_depth * filter_count +
                  //    in_channel * filter_count + out_channel;
                  // const T filter_value = filter(filter_index);
                  const T dw_filter_value =
                      dw_filter(filter_y, filter_x, in_channel, r);
                  const T pw_filter_value =
                      pw_filter(0, 0, in_channel*dw_filter_channel_mult + r, out_channel);
                  output_val += (input_value * dw_filter_value * pw_filter_value);
                }
              }
            }
          }

          /*
          out((batch * out_rows * out_cols * filter_count) +
                      (out_y * out_cols * filter_count) +
                      (out_x * filter_count) + out_channel) = output_val;
          */
          out(batch, out_y, out_x, out_channel) = output_val;
        }
      }
    }
  }
}

template <typename T>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, depthwise_filter, pointwise_filter };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  DepthwiseSeparableConvOperator(std::initializer_list<uint16_t> strides,
                                 Padding padding)
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape  = (*inputs[in].tensor)->get_shape();
    TensorShape& df_shape  = (*inputs[depthwise_filter].tensor)->get_shape();
    TensorShape& pf_shape  = (*inputs[pointwise_filter].tensor)->get_shape();
    TensorShape& out_shape = (*outputs[out].tensor)->get_shape();

    if (in_shape[3] != df_shape[2]){
      Context::get_default_context()->throwError(new InvalidTensorDimensionsError); 
    }
    if (pf_shape[0] != 1 || pf_shape[1] != 1) {
      Context::get_default_context()->throwError(new InvalidTensorDimensionsError); 
    }
    depthwise_separable_convolution_kernel<T>(*outputs[out].tensor, *inputs[in].tensor,
                          *inputs[depthwise_filter].tensor, 
                          *inputs[pointwise_filter].tensor, 
                          _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T, typename Filter>
class GenericPoolOp : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  GenericPoolOp(std::initializer_list<uint16_t> k_size, std::initializer_list<uint16_t> strides,
                                 Padding padding) 
      : _padding(padding) {
    int i = 0;
    for (auto s : strides) {
      _stride[i++] = s;
    }
    i = 0;
    for (auto k : k_size) {
      _k_size[i++] = k;
    }
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape  = (*inputs[in].tensor)->get_shape();
    Filter filter(_k_size[0], _k_size[1], in_shape[3]);
    generic_pool_convolution_kernel<T, Filter>(*outputs[out].tensor, *inputs[in].tensor,
                          filter,_padding, _stride);
  }

 private:
  uint16_t _k_size[2];
  uint16_t _stride[4];
  Padding _padding;
};

template<typename T>
using MaxPoolOp = GenericPoolOp<T, MaxFilter<T>>;

template<typename T>
using AvgPoolOp = GenericPoolOp<T, AvgFilter<T>>;

}  // namespace uTensor
#endif
