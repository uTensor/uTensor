#include "quantizationPrimitives.hpp"
#include "context.hpp"
#include "memoryManagementInterface.hpp"
#include "uTensor_util.hpp"

namespace uTensor {

DEFINE_ERROR(AttemptToUseUnSpecializedQuantizeParamsError);

QuantizationParams::QuantizationParams()
    : _zp(nullptr), _scale(nullptr), _num_channels(0) {}
QuantizationParams::QuantizationParams(const int32_t* zp, const float* scale,
                                       int num_channels)
    : _zp(zp), _scale(scale), _num_channels(num_channels) {}

QuantizationParams::~QuantizationParams() {}
int32_t QuantizationParams::get_zeroP_for_channel(int i) const { 
  uTensor_printf(
      "ERROR, Attempted to use non specialized Quantization params\n");
  Context::get_default_context()->throwError(
      new AttemptToUseUnSpecializedQuantizeParamsError());
  return 0; 
}
float QuantizationParams::get_scale_for_channel(int i) const { 
  uTensor_printf(
      "ERROR, Attempted to use non specialized Quantization params\n");
  Context::get_default_context()->throwError(
      new AttemptToUseUnSpecializedQuantizeParamsError());
  return 0; 
}
// Allocate the tensor metadata on a different heap from the data scratch pads
void* QuantizationParams::operator new(size_t sz) {
  void* p =
      Context::get_default_context()->get_metadata_allocator()->allocate(sz);
  return p;
}

void QuantizationParams::operator delete(void* p) {
  Context::get_default_context()->get_metadata_allocator()->deallocate(p);
}

PerTensorQuantizationParams::PerTensorQuantizationParams(const int32_t& zp,
                                                         const float& scale)
    : QuantizationParams(&zp, &scale, 1) {}
int32_t PerTensorQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp[0];
}
float PerTensorQuantizationParams::get_scale_for_channel(int i) const {
  return _scale[0];
}

int32_t PerChannelQuantizationParams::get_zeroP_for_channel(int i) const {
  return _zp[i];
}
float PerChannelQuantizationParams::get_scale_for_channel(int i) const {
  return _scale[i];
}

}  // namespace uTensor
