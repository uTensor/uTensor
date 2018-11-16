#include "supportOps.hpp"

void uint8_to_q7_origin(uint8_t *input, float &min, float &max, q7_t *out, size_t length) {
  //NT TODO: consider loop unrolling or SIMD here
  const int32_t offset = FloatToQuantizedUnclamped<uint8_t>(
      0.0f, min, max) - 128;
  for(size_t i = 0; i < length; i++) {
    out[i] = (q7_t) (input[i] + offset);
  }
}