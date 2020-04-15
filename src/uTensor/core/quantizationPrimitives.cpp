#include "quantizationPrimitives.hpp"

namespace uTensor {

QuantizationParams::QuantizationParams()
    : _zp(nullptr), _scale(nullptr), _num_channels(0) {}
QuantizationParams::QuantizationParams(const int32_t* zp, const float* scale,
                                       int num_channels)
    : _zp(zp), _scale(scale), _num_channels(num_channels) {}

QuantizationParams::~QuantizationParams() {}
int32_t const QuantizationParams::get_zeroP_for_channel(int i) const { return *_zp; }
float const QuantizationParams::get_scale_for_channel(int i) const { return *_scale; }

PerTensorQuantizationParams::PerTensorQuantizationParams(const int32_t& zp,
                                                         const float& scale)
    : QuantizationParams(&zp, &scale, 1) {}
int32_t const PerTensorQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp[0];
}
float const PerTensorQuantizationParams::get_scale_for_channel(int i) const {
  return _scale[0];
}

int32_t const PerChannelQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp[i];
}
float const PerChannelQuantizationParams::get_scale_for_channel(int i) const {
  return _scale[i];
}

}  // namespace uTensor
