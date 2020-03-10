#include "quantizationPrimitives.hpp"

namespace uTensor {

QuantizationParams::QuantizationParams()
    : _zp(nullptr), _scale(nullptr), _num_channels(0) {}
QuantizationParams::QuantizationParams(const int32_t* zp, const int32_t* scale,
                                       int num_channels)
    : _zp(zp), _scale(scale), _num_channels(num_channels) {}

QuantizationParams::~QuantizationParams() {}
int32_t QuantizationParams::get_zeroP_for_channel(int i) { return 0; }
int32_t QuantizationParams::get_scale_for_channel(int i) { return 0; }

PerTensorQuantizationParams::PerTensorQuantizationParams(const int32_t& zp,
                                                         const int32_t& scale)
    : QuantizationParams(&zp, &scale, 1) {}
int32_t PerTensorQuantizationParams::get_zeroP_for_channel(int i) {
  return _zp[0];
}
int32_t PerTensorQuantizationParams::get_scale_for_channel(int i) {
  return _scale[0];
}

int32_t PerChannelQuantizationParams::get_zeroP_for_channel(int i) {
  return _zp[i];
}
int32_t PerChannelQuantizationParams::get_scale_for_channel(int i) {
  return _scale[i];
}

}  // namespace uTensor
