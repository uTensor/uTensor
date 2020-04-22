#ifndef UTENSOR_CONVOLUTION_OPS_H
#define UTENSOR_CONVOLUTION_OPS_H
#include <algorithm>
#include <limits>

#include "Convolution_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {

// Can use these intermediate types to make the convolution operator more
// generic. Maxpool, conv, average pool, median etc. are all basically the same
// operation with target functions.
template <typename T>
class ConvFilter {
  T tmp;
  const Tensor& filter;

 public:
  ConvFilter(const Tensor& filter) : filter(filter) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    const T filter_value = filter(i, j, k, l);
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
class MinFilter {
  T tmp;
  int16_t h;
  int16_t w;
  int16_t c;

 public:
  MinFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = std::numeric_limits<T>::max(); }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp = std::min(tmp, input_value);
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
  AvgFilter(int16_t h, int16_t w, int16_t c) : h(h), w(w), c(c) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    tmp += input_value;
  }
  inline T finalize() const {
    return tmp / (w * h);
  }  //(static_cast<T>(w*h)); }
  inline const int16_t height() const { return h; }
  inline const int16_t width() const { return w; }
  inline const int16_t in_channels() const { return 1; }
  inline const int16_t out_channels() const { return c; }
};

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
    ConvFilter<T> conv(inputs[filter].tensor());
    generic_convolution_kernel<T, ConvFilter<T>>(
        outputs[out].tensor(), inputs[in].tensor(), conv, _padding, _stride);
  }

 private:
  uint16_t _stride[4];
  Padding _padding;
};

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
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    TensorShape& df_shape = inputs[depthwise_filter].tensor()->get_shape();
    TensorShape& pf_shape = inputs[pointwise_filter].tensor()->get_shape();
    TensorShape& out_shape = outputs[out].tensor()->get_shape();

    if (in_shape[3] != df_shape[2]) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    if (pf_shape[0] != 1 || pf_shape[1] != 1) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    depthwise_separable_convolution_kernel<T>(
        outputs[out].tensor(), inputs[in].tensor(),
        inputs[depthwise_filter].tensor(), inputs[pointwise_filter].tensor(),
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
  GenericPoolOp(std::initializer_list<uint16_t> k_size,
                std::initializer_list<uint16_t> strides, Padding padding)
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
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    Filter filter(_k_size[0], _k_size[1], in_shape[3]);
    generic_pool_convolution_kernel<T, Filter>(
        outputs[out].tensor(), inputs[in].tensor(), filter, _padding, _stride);
  }

 private:
  uint16_t _k_size[2];
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
using MaxPoolOp = GenericPoolOp<T, MaxFilter<T>>;

template <typename T>
using AvgPoolOp = GenericPoolOp<T, AvgFilter<T>>;

namespace TFLM {


enum PaddingType : uint8_t { kNone, kSame, kValid };

struct PaddingValues {
  int16_t width;
  int16_t height;
  // offset is used for calculating "remaining" padding, for example, `width`
  // is 1 and `width_offset` is 1, so padding_left is 1 while padding_right is
  // 1 + 1 = 2.
  int16_t width_offset;
  // Same as width_offset except it's over the height dimension.
  int16_t height_offset;
};
struct DepthwiseParams {
  PaddingType padding_type;
  PaddingValues padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  const int32_t* output_multiplier_per_channel;
  const int32_t* output_shift_per_channel;
};

uint16_t MatchingDim(TensorShape s0, uint8_t i0, TensorShape s1, uint8_t i1) {
  assert(s0[i0] == s1[i1]);
  return s0[i0];
}

template <typename T>
void DCHECK_LE(T v0, T v1) {
  assert(v0 <= v1);
}

template <typename T>
void DCHECK_EQ(T v0, T v1) {
  assert(v0 == v1);
}

inline void DepthwiseConvPerChannel(
const DepthwiseParams& params, const int32_t* output_multiplier, const int32_t* output_shift,
Tensor& input, Tensor& filter, Tensor& bias, Tensor& output
) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TensorShape input_shape = input->get_shape();
  TensorShape filter_shape = filter->get_shape();
  TensorShape output_shape = output->get_shape();
  

  assert(input_shape.num_dims() == 4);
  assert(filter_shape.num_dims() == 4);
  assert(output_shape.num_dims() == 4);

  DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape[1];
  const int input_width = input_shape[2];
  const int input_depth = input_shape[3];
  const int filter_height = filter_shape[1];
  const int filter_width = filter_shape[2];
  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  DCHECK_EQ(bias->get_shape().get_linear_size(), (uint16_t) output_depth);

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // Zero padding by omitting the areas outside the image.
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);
                if (is_point_inside_image) {
                  // int32_t input_val = input_data[Offset(input_shape, batch, in_y,
                  //                                     in_x, in_channel)];
                  int32_t input_val = input(batch, in_y, in_x, in_channel);
                  // int32_t filter_val = filter_data[Offset(
                  //     filter_shape, 0, filter_y, filter_x, output_channel)];
                  int32_t filter_val = filter(batch, filter_y, filter_x, output_channel);
                  // Accumulate with 32 bits accumulator.
                  // In the nudging process during model quantization, we force
                  // real value of 0.0 be represented by a quantized value. This
                  // guarantees that the input_offset is a int8, even though it
                  // is represented using int32.
                  // int32 += int8 * (int8 - int8) so the highest value we can
                  // get from each accumulation is [-127, 127] * ([-128, 127] -
                  // [-128, 127]), which is [-32512, 32512]. log2(32512)
                  // = 14.98, which means we can accumulate at least 2^16
                  // multiplications without overflow. The accumulator is
                  // applied to a filter so the accumulation logic will hold as
                  // long as the filter size (filter_y * filter_x * in_channel)
                  // does not exceed 2^16, which is the case in all the models
                  // we have seen so far.
                  // TODO(jianlijianli): Add a check to make sure the
                  // accumulator depth is smaller than 2^16.
                  acc += filter_val * (input_val + input_offset);
                }
              }
            }
            //assuming bias data will always be provided
            acc += (int32_t) bias(output_channel);
            // acc = MultiplyByQuantizedMultiplier(
            //     acc, output_multiplier[output_channel],
            //     output_shift[output_channel]);

            //simplified MultiplyByQuantizedMultiplier, may introduce rounding error
            int left_shift = output_shift[output_channel] > 0 ? output_shift[output_channel] : 0;
            int right_shift = output_shift[output_channel] > 0 ? 0 : -output_shift[output_channel];
            acc = ((acc * (1 << left_shift)) * output_multiplier[output_channel]) >> right_shift;

            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output(batch, out_y, out_x,
                               output_channel) = static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}
template <typename T>
class DepthwiseSeparableConvOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

  DepthwiseSeparableConvOperator& set_params(DepthwiseParams&& _params, const int32_t&& _output_multiplier, const int32_t&& _output_shift) {
    params = _params;
    output_multiplier = _output_multiplier;
    output_shift = _output_shift;
    return *this;
  }

 protected:
  virtual void compute() {
    TensorShape& in_shape = inputs[in].tensor()->get_shape();
    TensorShape& df_shape = inputs[filter].tensor()->get_shape();
    TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
    TensorShape& out_shape = outputs[out].tensor()->get_shape();

    if (in_shape[3] != df_shape[2]) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }
    if (bias_shape[0] != 1 || bias_shape[1] != 1) {
      Context::get_default_context()->throwError(
          new InvalidTensorDimensionsError);
    }

    DepthwiseConvPerChannel( 
      params, &output_multiplier, &output_shift,
      inputs[in].tensor(), inputs[filter].tensor(), inputs[bias].tensor(), outputs[out].tensor()
    );
  }

private:
  DepthwiseParams params;
  int32_t output_multiplier;
  int32_t output_shift;
};

}  // namespace TFLM
}  // namespace uTensor
#endif
