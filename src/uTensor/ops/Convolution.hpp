#ifndef UTENSOR_CONVOLUTION_OPS_H
#define UTENSOR_CONVOLUTION_OPS_H
#include <algorithm>
#include <limits>

#include "Convolution_kernels.hpp"
#include "operatorBase.hpp"

namespace uTensor {
namespace ReferenceOperators {

namespace Conv2dConstants {
  // https://github.com/tensorflow/tensorflow/blob/28d1ad34bb59e3e1631b5807eebc46563ef3382c/tensorflow/lite/kernels/internal/reference/conv.h#L56-L57
  constexpr int input_batch_dim         = 0;
  constexpr int input_height_dim        = 0;
  constexpr int input_witdh_dim         = 0;
  constexpr int input_channel_dim       = 0;

  constexpr int filter_height_dim       = 1;
  constexpr int filter_width_dim        = 2;
  constexpr int filter_in_channels_dim  = 3;
  constexpr int filter_out_channels_dim = 0;

  constexpr int output_height_dim       = 1;
  constexpr int output_width_dim        = 2;
}

// Can use these intermediate types to make the convolution operator more
// generic. Maxpool, conv, average pool, median etc. are all basically the same
// operation with target functions.
template <typename T>
class ConvFilter {
  T tmp;
  const Tensor& filter;

 public:
  ConvFilter(const Tensor& filter) : tmp(0), filter(filter) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const T& input_value, int i, int j, int k, int l) {
    const T filter_value = filter(i, j, k, l);
    tmp += (input_value * filter_value);
  }
  inline T finalize() const { return tmp; }
  inline const int16_t height() const { return filter->get_shape()[Conv2dConstants::filter_height_dim]; }
  inline const int16_t width() const { return filter->get_shape()[Conv2dConstants::filter_width_dim]; }
  inline const int16_t in_channels() const { return filter->get_shape()[Conv2dConstants::filter_in_channels_dim]; }
  inline const int16_t out_channels() const { return filter->get_shape()[Conv2dConstants::filter_out_channels_dim]; }
};
// Specialization for quantization
template <>
class ConvFilter<int8_t> {
  float tmp;
  const Tensor& filter;

 public:
  ConvFilter(const Tensor& filter) : tmp(0), filter(filter) {}
  inline void reset() { tmp = 0; }
  inline void PartialCompute(const float& input_value, int i, int j, int k, int l) {
    const int32_t fv32 = static_cast<int32_t>(static_cast<int8_t>(filter(i, j, k, l)));
    const int32_t zp = filter->get_quantization_params().get_zeroP_for_channel(i);
    const float scale = filter->get_quantization_params().get_scale_for_channel(i);
    const float filter_value = (fv32 - zp)*scale;
    tmp += (input_value * filter_value);
  }
  inline float finalize() const { return tmp; }
  inline const int16_t height() const { return filter->get_shape()[Conv2dConstants::filter_height_dim]; }
  inline const int16_t width() const { return filter->get_shape()[Conv2dConstants::filter_width_dim]; }
  inline const int16_t in_channels() const { return filter->get_shape()[Conv2dConstants::filter_in_channels_dim]; }
  inline const int16_t out_channels() const { return filter->get_shape()[Conv2dConstants::filter_out_channels_dim]; }
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

template<typename T>
class NoBias {
  public:
    T operator()(int32_t i) { return 0; }
};

template<typename T>
class wBias {
  public:
    wBias(const Tensor& t) : t(t) {}
    T operator()(int32_t i) { return static_cast<T>(t(i)); }

  private:
    const Tensor& t;
};
template<>
class wBias<int8_t> {
  public:
    wBias(const Tensor& t) : t(t) {}
    float operator()(int32_t i) { 
      const int32_t b32 = static_cast<int32_t>(t(i));
      const float scale = t->get_quantization_params().get_scale_for_channel(i);
      const int32_t zp = t->get_quantization_params().get_zeroP_for_channel(i);
      return (b32 - zp)*scale;
    }

  private:
    const Tensor& t;
};

template <typename T>
class Conv2dOperator : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };
  Conv2dOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding) {
    for(int j = 0; j < 4; j++){
      _stride[j] = 1;
    }
    int i = 0;
    if(strides.size() == 2){
      i = 1; // Offset the stride loc
    }
    for (auto s : strides) {
      _stride[i++] = s;
    }
  }

 protected:
  virtual void compute();

 private:
  uint16_t _stride[4];
  Padding _padding;
};

template <typename T>
void Conv2dOperator<T>::compute() {
  bool have_bias = inputs.has(bias);
  ConvFilter<T> conv(inputs[filter].tensor());
  if(have_bias) {
    wBias<T> w_bias(inputs[bias].tensor());
    generic_convolution_kernel<T, ConvFilter<T>>(
      outputs[out].tensor(), inputs[in].tensor(), conv, w_bias, _padding, _stride);
  } else {
    NoBias<T> no_bias;
    generic_convolution_kernel<T, ConvFilter<T>>(
      outputs[out].tensor(), inputs[in].tensor(), conv, no_bias, _padding, _stride);

  }
}

// Specialization for symmetric quantization
template <>
void Conv2dOperator<int8_t>::compute();

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
class GenericPoolOperator : public OperatorInterface<1, 1> {
 public:
  enum names_in : uint8_t { in };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  GenericPoolOperator(std::initializer_list<uint16_t> k_size,
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
using MaxPoolOperator = GenericPoolOperator<T, MaxFilter<T>>;

template <typename T>
using AvgPoolOperator = GenericPoolOperator<T, AvgFilter<T>>;

template<typename T>
using MinPoolOperator = GenericPoolOperator<T, MinFilter<T>>;

}
}  // namespace uTensor
#endif
