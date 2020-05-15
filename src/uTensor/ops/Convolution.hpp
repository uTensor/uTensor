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

template <typename Tout>
class DepthwiseSeparableConvOperatorV2 : public OperatorInterface<3, 1> {
 public:
  enum names_in : uint8_t { in, filter, bias };
  enum names_out : uint8_t { out };

 public:
  DepthwiseSeparableConvOperatorV2();
  // TODO allow 4D bits later
  //DepthwiseSeparableConvOperatorV2(
  //    const uint16_t (&strides)[4], Padding padding,
  //    const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1});
  DepthwiseSeparableConvOperatorV2(
      const uint16_t (&strides)[2], Padding padding,
      const int depth_multiplier = 1, const uint16_t (&dialation)[2] = {1, 1});

 protected:
  virtual void compute();

 private:
  // TfLiteDepthwiseConvParams
  // Set by constructors
  uint16_t _stride[4];
  Padding _padding;
  int depth_multiplier;
  uint16_t _dialation[2];
};

template <typename Tout>
DepthwiseSeparableConvOperatorV2<
    Tout>::DepthwiseSeparableConvOperatorV2()
    : _stride{1, 1},
      _padding(SAME),
      depth_multiplier(1),
      _dialation{1, 1} {}

template <typename Tout>
DepthwiseSeparableConvOperatorV2<Tout>::
    DepthwiseSeparableConvOperatorV2(
        const uint16_t (&strides)[2], Padding padding,
        const int depth_multiplier, const uint16_t (&dialation)[2])
    : _stride{1, strides[0], strides[1], 1}, _padding(padding),
      depth_multiplier(depth_multiplier),
      _dialation{dialation[0], dialation[1]}
{
}

template <typename Tout>
void DepthwiseSeparableConvOperatorV2<Tout>::compute() {
  AllocatorInterface* ram_allocator =
      Context::get_default_context()->get_ram_data_allocator();
  const TensorShape& in_shape = inputs[in].tensor()->get_shape();
  const TensorShape& df_shape = inputs[filter].tensor()->get_shape();
  const TensorShape& bias_shape = inputs[bias].tensor()->get_shape();
  const TensorShape& out_shape = outputs[out].tensor()->get_shape();

  if (in_shape[3] != df_shape[2]) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }
  if (bias_shape[0] != 1 || bias_shape[1] != 1) {
    Context::get_default_context()->throwError(
        new InvalidTensorDimensionsError);
  }


  depthwise_separable_convolution_kernel_v2<Tout>(
      outputs[out].tensor(),
      inputs[in].tensor(), inputs[filter].tensor(), inputs[bias].tensor(),
      _padding, _stride, depth_multiplier, _dialation);

}


}  // namespace uTensor
#endif
