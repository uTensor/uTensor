#include "src/uTensor/util/quantization_utils.hpp"

void RequantizeManyInNewRangeReference(const int* input, int32_t count,
    float min_input, float max_input,
    float min_output,
    float max_output,
    unsigned char* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
      const int fp_shift = 16;
      const float input_range = max_input - min_input;
      const float output_range = max_output - min_output;
      const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
      const float input_rezero = (min_input + max_input) / 2.0;
      const int64_t range_scale_fp =
      output_range == 0.0 ? 0.0
      : static_cast<int64_t>(255.0 * (1 << fp_shift) *
        input_range / output_range);
      const int64_t input_offset_fp =
      static_cast<int64_t>(input_rezero * recip_output_range * (1 << fp_shift));
      const int64_t output_offset_fp =
      output_range == 0.0
      ? 0
      : static_cast<int64_t>((1 << fp_shift) * (min_output * 255.0) /
      output_range);
      const int64_t rounding_delta = 1 << (fp_shift - 1);

      // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
      // that could be easily adapted for a SIMD implementation. It should also be
      // possible to perform all the calculations in 32-bit rather than 64, but
      // that's not been implemented yet.
      for (int32_t index = 0; index < count; ++index) {
        const int64_t input_value = static_cast<int64_t>(input[index]);
        const int64_t fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
        const int64_t offset_intermediate = fp_value - output_offset_fp;
        const int64_t round_intermediate = offset_intermediate + rounding_delta;
        int64_t quantized_int64 = round_intermediate >> fp_shift;
        quantized_int64 = std::max(quantized_int64, (int64_t) 0);
        quantized_int64 = std::min(quantized_int64, (int64_t) 255);
        output[index] = static_cast<unsigned char>(static_cast<int32_t>(quantized_int64));
      }
}
