#ifndef UTENSOR_CMSIS_CONV_H
#define UTENSOR_CMSIS_CONV_H

#include <algorithm>
#include <limits>
#include "operatorBase.hpp"
#include "Convolution_kernels.hpp"

namespace uTensor {

template <typename T>
class CmsisConvOperator : public OperatorInterface<2, 1>, FastOperator {
 public:
  enum names_in : uint8_t { in, filter };
  enum names_out : uint8_t { out };
  CmsisConvOperator(std::initializer_list<uint16_t> strides, Padding padding)
      : _padding(padding), 
        output_activation_min(std::numeric_limits<int8_t>::min(),
        output_activation_max(std::numeric_limits<int8_t>::max() {
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
  // Do something with this eventually
  const int32_t output_activation_min;
  const int32_t output_activation_max;
};


template <typename T>
class CmsisDepthwiseSeparableConvOperator : public OperatorInterface<3, 1>, FastOperator {
 public:
  enum names_in : uint8_t { in, depthwise_filter, pointwise_filter };
  enum names_out : uint8_t { out };

  // TODO Add dialations
  CmsisDepthwiseSeparableConvOperator(std::initializer_list<uint16_t> strides,
                                 Padding padding)
      : _padding(padding),
        output_activation_min(std::numeric_limits<int8_t>::min(),
        output_activation_max(std::numeric_limits<int8_t>::max() {
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
  // Do something with this eventually
  const int32_t output_activation_min;
  const int32_t output_activation_max;
};


}
#endif
