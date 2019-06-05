#include "MathOps.hpp"

// Place full specializations here
// QuantizedAdd specialization
// https://github.com/tensorflow/tensorflow/blob/d0a5d88/tensorflow/core/kernels/quantized_add_op.cc#L245
template <>
void QuantizedAdd<uint8_t, uint8_t, int>(
  S_TENSOR input_x, S_TENSOR input_y,
  S_TENSOR min_x, S_TENSOR max_x,
  S_TENSOR min_y, S_TENSOR max_y,
  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {

  if (!output->getSize()) output->resize(input_x->getShape());

  const float x_min_float = *(min_x->read<float>(0, 0));
  const float x_max_float = *(max_x->read<float>(0, 0));
  const float y_min_float = *(min_y->read<float>(0, 0));
  const float y_max_float = *(max_y->read<float>(0, 0));

  const float smaller_min = std::min(x_min_float, y_min_float);
  const float larger_max = std::max(x_max_float, y_max_float);
  const float biggest_range = std::max(std::abs(smaller_min), std::abs(larger_max));
  const float output_range = (biggest_range * (1 << 14));
  const float output_min_float = -output_range;
  const float output_max_float = output_range;
  *(out_min->write<float>(0, 0)) = output_min_float;
  *(out_max->write<float>(0, 0)) = output_max_float;

  float x_0_float = QuantizedToFloat<uint8_t>(0, x_min_float, x_max_float);
  float x_1_float = QuantizedToFloat<uint8_t>(1, x_min_float, x_max_float);
  const int64_t x_0_int64 = FloatToQuantizedUnclamped<int32_t>(x_0_float, output_min_float, output_max_float);
  const int64_t x_1_int64 = FloatToQuantizedUnclamped<int32_t>(x_1_float, output_min_float, output_max_float);
  const int32_t x_mult_int32 = x_1_int64 - x_0_int64;

  float y_0_float = QuantizedToFloat<uint8_t>(0, y_min_float, y_max_float);
  float y_1_float = QuantizedToFloat<uint8_t>(1, y_min_float, y_max_float);
  const int64_t y_0_int64 = FloatToQuantizedUnclamped<int32_t>(y_0_float, output_min_float, output_max_float);
  const int64_t y_1_int64 = FloatToQuantizedUnclamped<int32_t>(y_1_float, output_min_float, output_max_float);
  const int32_t y_mult_int32 = y_1_int64 - y_0_int64;

  const int64_t quant_lowest = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
  const int64_t quant_highest = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

  uint32_t input_elems_cnt = input_x->getSize();
  uint32_t smaller_elems_cnt = input_y->getSize();
  const uint8_t *ptr_x = input_x->read<uint8_t>(0, 0);
  const uint8_t *ptr_y = input_y->read<uint8_t>(0, 0);
  int *ptr_out = output->write<int>(0, 0);
  for (size_t off = 0; off < input_elems_cnt; ++off) {
    size_t idx = off % smaller_elems_cnt;

    int32_t x_value_32 = static_cast<int32_t>(ptr_x[off]);
    int64_t x_in_output_64 = x_0_int64 + static_cast<int64_t>(x_value_32*x_mult_int32);
    x_in_output_64 = std::max(x_in_output_64, quant_lowest);
    x_in_output_64 = std::min(x_in_output_64, quant_highest);
    const int x_in_output_range = static_cast<int>(x_in_output_64);
    
    int32_t y_value_32 = static_cast<int32_t>(ptr_y[idx]);
    int64_t y_in_output_64 = y_0_int64 + static_cast<int64_t>(y_value_32*y_mult_int32);
    y_in_output_64 = std::max(y_in_output_64, quant_lowest);
    y_in_output_64 = std::min(y_in_output_64, quant_highest);
    const int y_in_output_range = static_cast<int>(y_in_output_64);

    ptr_out[off] = x_in_output_range + y_in_output_range;
  }
}
/*
template <>
void QuantizedMul<uint8_t, uint8_t, int>(
  S_TENSOR input_x, S_TENSOR input_y,
  S_TENSOR min_x, S_TENSOR max_x,
  S_TENSOR min_y, S_TENSOR max_y,
  S_TENSOR output, S_TENSOR out_min, S_TENSOR out_max) {

  if (!output->getSize()) output->resize(input_x->getShape());

  const float x_min_float = *(min_x->read<float>(0, 0));
  const float x_max_float = *(max_x->read<float>(0, 0));
  const float y_min_float = *(min_y->read<float>(0, 0));
  const float y_max_float = *(max_y->read<float>(0, 0));

  const float smaller_min = std::min(x_min_float, y_min_float);
  const float larger_max = std::max(x_max_float, y_max_float);
  const float biggest_range = std::max(std::abs(smaller_min), std::abs(larger_max));
  const float output_range = (biggest_range * (1 << 14));
  const float output_min_float = -output_range;
  const float output_max_float = output_range;
  *(out_min->write<float>(0, 0)) = output_min_float;
  *(out_max->write<float>(0, 0)) = output_max_float;

  float x_0_float = QuantizedToFloat<uint8_t>(0, x_min_float, x_max_float);
  float x_1_float = QuantizedToFloat<uint8_t>(1, x_min_float, x_max_float);
  const int64_t x_0_int64 = FloatToQuantizedUnclamped<int32_t>(x_0_float, output_min_float, output_max_float);
  const int64_t x_1_int64 = FloatToQuantizedUnclamped<int32_t>(x_1_float, output_min_float, output_max_float);
  const int32_t x_mult_int32 = x_1_int64 - x_0_int64;

  float y_0_float = QuantizedToFloat<uint8_t>(0, y_min_float, y_max_float);
  float y_1_float = QuantizedToFloat<uint8_t>(1, y_min_float, y_max_float);
  const int64_t y_0_int64 = FloatToQuantizedUnclamped<int32_t>(y_0_float, output_min_float, output_max_float);
  const int64_t y_1_int64 = FloatToQuantizedUnclamped<int32_t>(y_1_float, output_min_float, output_max_float);
  const int32_t y_mult_int32 = y_1_int64 - y_0_int64;

  const int64_t quant_lowest = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
  const int64_t quant_highest = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

  uint32_t input_elems_cnt = input_x->getSize();
  uint32_t smaller_elems_cnt = input_y->getSize();
  const uint8_t *ptr_x = input_x->read<uint8_t>(0, 0);
  const uint8_t *ptr_y = input_y->read<uint8_t>(0, 0);
  int *ptr_out = output->write<int>(0, 0);
  for (size_t off = 0; off < input_elems_cnt; ++off) {
    size_t idx = off % smaller_elems_cnt;

    int32_t x_value_32 = static_cast<int32_t>(ptr_x[off]);
    int64_t x_in_output_64 = x_0_int64 + static_cast<int64_t>(x_value_32*x_mult_int32);
    x_in_output_64 = std::max(x_in_output_64, quant_lowest);
    x_in_output_64 = std::min(x_in_output_64, quant_highest);
    const int x_in_output_range = static_cast<int>(x_in_output_64);
    
    int32_t y_value_32 = static_cast<int32_t>(ptr_y[idx]);
    int64_t y_in_output_64 = y_0_int64 + static_cast<int64_t>(y_value_32*y_mult_int32);
    y_in_output_64 = std::max(y_in_output_64, quant_lowest);
    y_in_output_64 = std::min(y_in_output_64, quant_highest);
    const int y_in_output_range = static_cast<int>(y_in_output_64);

    ptr_out[off] = x_in_output_range * y_in_output_range;
  }
}
*/