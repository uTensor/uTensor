#ifndef UTENSOR_QUANT_PRIM_H
#define UTENSOR_QUANT_PRIM_H
#include "types.hpp"

namespace uTensor {

class QuantizationParams {
 public:
  QuantizationParams();
  QuantizationParams(const int32_t* zp, const int32_t* scale, int num_channels);
  // TODO Move to cpp file
  QuantizationParams(const QuantizationParams&) = default;
  QuantizationParams& operator=(const QuantizationParams&) = default;
  QuantizationParams(QuantizationParams&&) = default;
  QuantizationParams& operator=(QuantizationParams&&) = default;

  virtual ~QuantizationParams();
  virtual int32_t get_zeroP_for_channel(int i);
  virtual int32_t get_scale_for_channel(int i);
  inline const int num_channels() const { return _num_channels; };

 protected:
  const int32_t* _zp;
  const int32_t* _scale;
  int _num_channels;
};

using NotQuantized = QuantizationParams;

inline bool is_quantized(const QuantizationParams& params) {
  return params.num_channels() > 0;
}
inline bool is_per_channel_quantization(const QuantizationParams& params) {
  return params.num_channels() > 1;
}

class PerTensorQuantizationParams : public QuantizationParams {
 public:
  PerTensorQuantizationParams(const int32_t& zp, const int32_t& scale);
  virtual int32_t get_zeroP_for_channel(int i) override;
  virtual int32_t get_scale_for_channel(int i) override;
  // virtual inline const int num_channels() { return _num_channels; };
};

class PerChannelQuantizationParams : public QuantizationParams {
 public:
  // PerChannelQuantizationParams(const int32_t* zp, const int32_t* scale, int
  // num_channels) : _zp(zp), _scale(scale), _num_channels(num_channels) {}

  // Smarter compile time thing might be
  template <int num_channels>
  PerChannelQuantizationParams(const int32_t (&zp)[num_channels],
                               const int32_t (&scale)[num_channels])
      : QuantizationParams(zp, scale, num_channels) {}

  virtual int32_t get_zeroP_for_channel(int i) override;
  virtual int32_t get_scale_for_channel(int i) override;
};

}  // namespace uTensor

#endif
