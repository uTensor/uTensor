#include "supportOps.hpp"

void uint8_to_q7_origin(const uint8_t *input, float &min, float &max, q7_t *out, size_t length) {
  
  const int32_t offset = FloatToQuantizedUnclamped<uint8_t>(
       0.0f, min, max);

  //printf("offset: %d", offset);
  for(size_t i = 0; i < length; i++) {
    out[i] = (q7_t) ((input[i] - offset) >> 1) ;
    //if(i < 16) printf("(%d: %d, %d), ", input[i], ((q7_t) ((input[i] - offset) >> 1)), out[i]);
  }
  
  //FIXME: the following code disabled for testing input shift method
  /*
  //NT TODO: consider loop unrolling or SIMD here
  const int32_t offset = FloatToQuantizedUnclamped<uint8_t>(
      0.0f, min, max) + 128;
  for(size_t i = 0; i < length; i++) {
    out[i] = (q7_t) (input[i] - offset);
  }
  */
}

template <typename TIn, typename TOut>
void right_shift(const TIn *input, TOut *out, const uint8_t shift, const size_t length) {
  for(size_t i = 0; i < length; i++) {
    out[i] = (TOut) (input[i]  << shift) ;
  }
}