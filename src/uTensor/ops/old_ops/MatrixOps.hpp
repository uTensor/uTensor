#ifndef UTENSOR_MATRIX_OPS
#define UTENSOR_MATRIX_OPS

#include "src/uTensor/util/quantization_utils.hpp"
#include "src/uTensor/core/tensor.hpp"
#include "src/uTensor/core/uTensorBase.hpp"
//#include "uTensor/ops/quantization.hpp" // Deprecated
#include <cmath>
#include <cstdlib>
#include <limits>

// tensorflow/tensorflow/core/kernels/reference_gemm.h

template <class T1, class T2, class T3>
void ReferenceGemmuImpl(bool transpose_a, bool transpose_b, bool transpose_c,
                        size_t m, size_t n, size_t k, S_TENSOR a,
                        int32_t offset_a, size_t lda, S_TENSOR b, int offset_b,
                        size_t ldb, S_TENSOR c, int shift_c, int offset_c,
                        int mult_c, size_t ldc) {
  int a_i_stride = lda;
  int a_l_stride = 1;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  }

  int b_j_stride = 1;
  int b_l_stride = ldb;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  }

  int c_i_stride = ldc;
  int c_j_stride = 1;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  }

  const int32_t highest = static_cast<int32_t>(std::numeric_limits<T3>::max());
  const int32_t lowest = static_cast<int32_t>(std::numeric_limits<T3>::min());
  const int32_t rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  size_t i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const T1* a_data = a->read<T1>(a_index, 1);
        const int32_t a_value = static_cast<int32_t>(a_data[0]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const T2* b_data = b->read<T2>(b_index, 1);
        const int32_t b_value = static_cast<int32_t>(b_data[0]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      T3* c_data = c->write<T3>(c_index, 1);
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }

      if (output < lowest) {
        output = lowest;
      }
      c_data[0] = static_cast<T3>(output);
    }
  }
}


template <class T1, class T2, class Toutput>
void QuantizedMatMul(Tensor* A, Tensor* B, Tensor** C,
                     Tensor* mina, Tensor* minb, Tensor* maxa,
                     Tensor* maxb, Tensor* outmin,
                     Tensor* outmax, bool transpose_a = false,
                     bool transpose_b = false) {
  const float min_a = *(mina->read<float>(0, 0));
  const float max_a = *(maxa->read<float>(0, 0));
  const float min_b = *(minb->read<float>(0, 0));
  const float max_b = *(maxb->read<float>(0, 0));

  //auto tensor allocation
  TensorShape c_shape;
  c_shape.push_back((A->getShape())[0]);
  c_shape.push_back((B->getShape())[1]);
  tensorChkAlloc<Toutput>(C, c_shape);

  const int32_t offset_a = FloatToQuantizedUnclamped<T1>(
      0.0f, min_a, max_a);  // NT: what 0 quantized to; depends on
                            // Eigen::NumTraits<T>::lowest()
  const int32_t offset_b = FloatToQuantizedUnclamped<T2>(0.0f, min_b, max_b);
  const int32_t offset_c = 0;
  const int32_t mult_c = 1;
  const int32_t shift_c = 0;

  int first = transpose_a ? 0 : 1;
  int second = transpose_b ? 1 : 0;

  int a_dim_remaining = 1 - first;
  int b_dim_remaining = 1 - second;

  const T1* A_Data = A->read<T1>(0, 0);
  const T2* B_Data = B->read<T2>(0, 0);
  Toutput* C_Data = (*C)->write<Toutput>(0, 0);

  const bool transpose_c = false;
  const size_t m = A->getShape()[a_dim_remaining];
  const size_t n = B->getShape()[b_dim_remaining];
  const size_t k = A->getShape()[first];
  const size_t lda = A->getShape()[1];
  const size_t ldb = B->getShape()[1];
  const size_t ldc = n;

  ReferenceGemmuImpl<T1, T2, Toutput>(
      transpose_a, transpose_b, transpose_c, m, n, k, A_Data, offset_a, lda,
      B_Data, offset_b, ldb, C_Data, shift_c, offset_c, mult_c, ldc);
  float min_c_value;
  float max_c_value;

  QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);

  float* c_min = outmin->write<float>(0, 0);
  *c_min = min_c_value;
  float* c_max = outmax->write<float>(0, 0);
  *c_max = max_c_value;
}

//////////////////////////////////////////////////////
template <class T1, class T2, class Toutput>
void MatMul2(S_TENSOR A, S_TENSOR B, S_TENSOR C,
                     bool transpose_a = false,
                     bool transpose_b = false) {

  //auto tensor allocation
  if(C->getSize() == 0) {
    TensorShape c_shape;
    c_shape.push_back((A->getShape())[0]);
    c_shape.push_back((B->getShape())[1]);
    C->resize(c_shape);
  }

  int first = transpose_a ? 0 : 1;
  int second = transpose_b ? 1 : 0;

  int a_dim_remaining = 1 - first;
  int b_dim_remaining = 1 - second;

  const bool transpose_c = false;
  const size_t m = A->getShape()[a_dim_remaining];
  const size_t n = B->getShape()[b_dim_remaining];
  const size_t k = A->getShape()[first];
  const size_t lda = A->getShape()[1];
  const size_t ldb = B->getShape()[1];
  const size_t ldc = n;

  int a_i_stride = lda;
  int a_l_stride = 1;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  }

  int b_j_stride = 1;
  int b_l_stride = ldb;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  }

  int c_i_stride = ldc;
  int c_j_stride = 1;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  }

  size_t i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      Toutput output = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const T1* a_value = A->read<T1>(a_index, 1);
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const T2* b_value = B->read<T2>(b_index, 1);
        output += (a_value[0] * b_value[0]);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      Toutput* c_data = C->write<Toutput>(c_index, 1);
      c_data[0] = static_cast<Toutput>(output);
    }
  }
}

template <class T1, class T2, class Toutput>
void QuantizedMatMul2(S_TENSOR A, S_TENSOR B, S_TENSOR C,
                     S_TENSOR mina, S_TENSOR minb, S_TENSOR maxa,
                     S_TENSOR maxb, S_TENSOR outmin,
                     S_TENSOR outmax, bool transpose_a = false,
                     bool transpose_b = false) {
  const float min_a = *(mina->read<float>(0, 0));
  const float max_a = *(maxa->read<float>(0, 0));
  const float min_b = *(minb->read<float>(0, 0));
  const float max_b = *(maxb->read<float>(0, 0));

  //auto tensor allocation
  if(C->getSize() == 0) {
    TensorShape c_shape;
    c_shape.push_back((A->getShape())[0]);
    c_shape.push_back((B->getShape())[1]);
    C->resize(c_shape);
  }
  

  const int32_t offset_a = FloatToQuantizedUnclamped<T1>(
      0.0f, min_a, max_a);  // NT: what 0 quantized to; depends on
                            // Eigen::NumTraits<T>::lowest()
  const int32_t offset_b = FloatToQuantizedUnclamped<T2>(0.0f, min_b, max_b);
  const int32_t offset_c = 0;
  const int32_t mult_c = 1;
  const int32_t shift_c = 0;

  int first = transpose_a ? 0 : 1;
  int second = transpose_b ? 1 : 0;

  int a_dim_remaining = 1 - first;
  int b_dim_remaining = 1 - second;

  const bool transpose_c = false;
  const size_t m = A->getShape()[a_dim_remaining];
  const size_t n = B->getShape()[b_dim_remaining];
  const size_t k = A->getShape()[first];
  const size_t lda = A->getShape()[1];
  const size_t ldb = B->getShape()[1];
  const size_t ldc = n;

  ReferenceGemmuImpl<T1, T2, Toutput>(
      transpose_a, transpose_b, transpose_c, m, n, k, A, offset_a, lda,
      B, offset_b, ldb, C, shift_c, offset_c, mult_c, ldc);
  float min_c_value;
  float max_c_value;

  QuantizationRangeForMultiplication<T1, T2, Toutput>(
      min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);

  float* c_min = outmin->write<float>(0, 0);
  *c_min = min_c_value;
  float* c_max = outmax->write<float>(0, 0);
  *c_max = max_c_value;
}

template<class T1, class T2, class T3>
void conv_functor(S_TENSOR input_data, int input_batches, int input_height, int input_width,
        int input_depth, S_TENSOR filter_data, int filter_height, int filter_width,
        int filter_count, int stride_rows, int stride_cols, Padding padding, S_TENSOR output_data,
        int output_height, int output_width)
{

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different positions in the input.
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          // Each filter kernel produces one output channel.
          for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
            const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
            const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
            T3 output_val = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  T1 input_value;
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                      size_t input_index = batch * input_height * input_width * input_depth +
                          in_y * input_width * input_depth + in_x * input_depth + in_channel;
                    const T1 *input_source_ptr =
                        input_data->template read<T1>(input_index, 1);
                    input_value = input_source_ptr[0];
                  } else {
                    input_value = 0;
                  }
                  size_t filter_index = filter_y * filter_width * input_depth * filter_count +
                      filter_x * input_depth * filter_count +
                      in_channel * filter_count + out_channel;
                  const T2 *filter_ptr =
                      filter_data->template read<T2>(filter_index, 1);
                  const T2 filter_value = filter_ptr[0];
                  output_val += (input_value * filter_value);
                }
              }
            }

            T3 *output =
            output_data->template write<T3>((batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel, 1);
            output[0] = static_cast<T3>(output_val);
          }
        }
      }
    }
}

template<class T1, class T2, class T3>
void fused_conv_maxpool_functor(S_TENSOR input_data, int input_batches, int input_height, int input_width,
        int input_depth, S_TENSOR filter_data, int filter_height, int filter_width,
        int filter_count, int stride_rows, int stride_cols, Padding padding, S_TENSOR output_data,
        int output_height, int output_width)
{

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    int filter_left_offset;
    int filter_top_offset;

    int max_pool_width = input_width/output_width;
    int max_pool_height = input_height/output_height;

    //Need to make sure these padding values are still valid
    if (padding == VALID) {
      filter_left_offset =
          //((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          (((input_width - filter_width)/stride_cols) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = (((input_height - filter_height)/stride_rows) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((input_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((input_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different positions in the input.
      // Move channel loop to the outside
      for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
        // For each max pooled value
        for (int max_out_y = 0; max_out_y < output_height; ++max_out_y) {
          for (int max_out_x = 0; max_out_x < output_width; ++max_out_x) {
            //Expand the max pool
            T3 max_pool_value = std::numeric_limits<T3>::lowest();
            for(int max_i_y = 0; max_i_y < max_pool_height; ++max_i_y){
              for(int max_i_x = 0; max_i_x < max_pool_width; ++max_i_x){
                //Run the conv filter
                int out_x = max_out_x*max_pool_width + max_i_x;
                int out_y = max_out_y*max_pool_height + max_i_y;
                // Each filter kernel produces one output channel.
                const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
                const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
                T3 output_val = 0;
                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    for (int in_channel = 0; in_channel < input_depth;
                         ++in_channel) {
                      const int in_x = in_x_origin + filter_x;
                      const int in_y = in_y_origin + filter_y;
                      T1 input_value;
                      if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                          (in_y < input_height)) {
                          size_t input_index = batch * input_height * input_width * input_depth +
                              in_y * input_width * input_depth + in_x * input_depth + in_channel;
                        const T1 *input_source_ptr =
                            input_data->template read<T1>(input_index, 1);
                        input_value = input_source_ptr[0];
                      } else {
                        input_value = 0;
                      }
                      size_t filter_index = filter_y * filter_width * input_depth * filter_count +
                          filter_x * input_depth * filter_count +
                          in_channel * filter_count + out_channel;
                      const T2 *filter_ptr =
                          filter_data->template read<T2>(filter_index, 1);
                      const T2 filter_value = filter_ptr[0];
                      output_val += (input_value * filter_value);
                    }
                  }
                }// End filter val
                max_pool_value = std::max(max_pool_value, output_val);
              }
            }
                
            T3 *output =
            output_data->template write<T3>((batch * output_height * output_width * filter_count) +
                        (max_out_y * output_width * filter_count) +
                        (max_out_x * filter_count) + out_channel, 1);
            output[0] = static_cast<T3>(max_pool_value);
          }
        }
      }
    }
}

template<class T1, class T2, class T3>
void quant_fused_conv_maxpool_functor(S_TENSOR input_data, int input_batches, int input_height, int input_width,
        int input_depth, int input_offset, S_TENSOR filter_data, int filter_height, int filter_width,
        int filter_count, int filter_offset, int stride_rows, int stride_cols, Padding padding, S_TENSOR output_data,
        int output_height, int output_width, int output_shift, int output_offset, int output_mult)
{

    const int32_t highest = static_cast<int32_t>(std::numeric_limits<T3>::max());
    const int32_t lowest = static_cast<int32_t>(std::numeric_limits<T3>::lowest());
    
    const int32_t rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1)); 

    // When we're converting the 32 bit accumulator to a lower bit depth, we
    int filter_left_offset;
    int filter_top_offset;

    int max_pool_width = input_width/output_width;
    int max_pool_height = input_height/output_height;

    //Need to make sure these padding values are still valid
    if (padding == VALID) {
      filter_left_offset =
          //((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          (((input_width - filter_width)/stride_cols) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = (((input_height - filter_height)/stride_rows) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((input_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((input_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different positions in the input.
      // Move channel loop to the outside
      for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
        // For each max pooled value
        for (int max_out_y = 0; max_out_y < output_height; ++max_out_y) {
          for (int max_out_x = 0; max_out_x < output_width; ++max_out_x) {
            //Expand the max pool
            int32_t max_pool_value = std::numeric_limits<int32_t>::lowest();
            for(int max_i_y = 0; max_i_y < max_pool_height; ++max_i_y){
              for(int max_i_x = 0; max_i_x < max_pool_width; ++max_i_x){
                //Run the conv filter
                int out_x = max_out_x*max_pool_width + max_i_x;
                int out_y = max_out_y*max_pool_height + max_i_y;
                // Each filter kernel produces one output channel.
                const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
                const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
                int32_t output_val = 0;
                for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                  for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    for (int in_channel = 0; in_channel < input_depth;
                         ++in_channel) {
                      const int in_x = in_x_origin + filter_x;
                      const int in_y = in_y_origin + filter_y;
                      int32_t input_value;
                      if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                          (in_y < input_height)) {
                          size_t input_index = batch * input_height * input_width * input_depth +
                              in_y * input_width * input_depth + in_x * input_depth + in_channel;
                        const T1 *input_source_ptr =
                            input_data->template read<T1>(input_index, 1);
                        input_value = static_cast<int32_t>(input_source_ptr[0]) - input_offset;
                      } else {
                        input_value = 0;
                      }
                      size_t filter_index = filter_y * filter_width * input_depth * filter_count +
                          filter_x * input_depth * filter_count +
                          in_channel * filter_count + out_channel;
                      const T2 *filter_ptr =
                          filter_data->template read<T2>(filter_index, 1);
                      const int32_t filter_value = 
                          static_cast<int32_t>(filter_ptr[0]) - filter_offset;
                      output_val += (input_value * filter_value);
                    }
                  }
                }// End filter val
                //Finish quant stuffs then maxpool it
                output_val =
                    ((((output_val + output_offset) * output_mult) + rounding) >>
                    output_shift);
                int32_t top_clamped_output = std::min(output_val, highest);
                int32_t clamped_output = std::max(top_clamped_output, lowest);
                max_pool_value = std::max(max_pool_value, clamped_output);
              }
            }
                
            T3 *output =
            output_data->template write<T3>((batch * output_height * output_width * filter_count) +
                        (max_out_y * output_width * filter_count) +
                        (max_out_x * filter_count) + out_channel, 1);
            output[0] = static_cast<T3>(max_pool_value);
          }
        }
      }
    }
}

template<class T1, class T2, class T3>
void quant_conv_functor(S_TENSOR input_data, int input_batches, int input_height, int input_width,
        int input_depth, int input_offset, S_TENSOR filter_data, int filter_height, int filter_width, 
        int filter_count, int filter_offset, int stride_rows, int stride_cols, Padding padding, S_TENSOR output_data,
        int output_height, int output_width, int output_shift, int output_offset, int output_mult) 
{
    const int32_t highest = static_cast<int32_t>(std::numeric_limits<T3>::max());
    const int32_t lowest = static_cast<int32_t>(std::numeric_limits<T3>::lowest());
    
    const int32_t rounding = (output_shift < 1) ? 0 : (1 << (output_shift - 1)); 


    // When we're converting the 32 bit accumulator to a lower bit depth, we
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < input_batches; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different positions in the input.
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          // Each filter kernel produces one output channel.
          for (int out_channel = 0; out_channel < filter_count; ++out_channel) {
            const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
            const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
            int32_t total = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int in_channel = 0; in_channel < input_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  int32_t input_value;
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                      size_t input_index = batch * input_height * input_width * input_depth +
                          in_y * input_width * input_depth + in_x * input_depth + in_channel;
                    const T1 *input_source_ptr =  
                        input_data->template read<T1>(input_index, 1);
                    input_value = static_cast<int32_t>(input_source_ptr[0]) - input_offset;
                  } else {
                    input_value = 0;
                  }
                  size_t filter_index = filter_y * filter_width * input_depth * filter_count +
                      filter_x * input_depth * filter_count +
                      in_channel * filter_count + out_channel;
                  const T2 *filter_ptr =
                      filter_data->template read<T2>(filter_index, 1);
                  const int32_t filter_value = 
                      static_cast<int32_t>(filter_ptr[0]) - filter_offset;
                  total += (input_value * filter_value);
                }
              }
            }
            int32_t output_val =
                ((((total + output_offset) * output_mult) + rounding) >>
                 output_shift);
            // We need to saturate the results against the largest and smallest
            
            int32_t top_clamped_output = std::min(output_val, highest);
            int32_t clamped_output = std::max(top_clamped_output, lowest);
            T3 *output = 
            output_data->template write<T3>((batch * output_height * output_width * filter_count) +
                        (out_y * output_width * filter_count) +
                        (out_x * filter_count) + out_channel, 1);
            *output = static_cast<T3>(clamped_output);
          }
        }
      }
    }
}

// For now maxpool strides are ignored and assumed to be on tap boundaries
template <class T1, class T2, class Toutput>
void FusedConvMaxPool(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   std::vector<int32_t> strides_, std::vector<int32_t> ksize, Padding padding_) {
  const int32_t in_depth = input->getShape()[3];
  const int32_t out_depth = filter->getShape()[3];
  const int32_t input_rows = input->getShape()[1];
  const int32_t filter_rows = filter->getShape()[0];
  const int32_t max_pool_width = ksize[1];

  const int32_t input_cols = input->getShape()[2];
  const int32_t filter_cols = filter->getShape()[1];
  const int32_t batch = input->getShape()[0];
  const int32_t max_pool_height = ksize[2];

  const int stride_rows = strides_[0];
  const int stride_cols = strides_[1];

  int32_t out_rows, out_cols;
  if (padding_ == VALID) {
    out_rows = (input_rows - filter_rows) / stride_rows + 1;
    out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    out_rows = input_rows;
    out_cols = input_cols;
  }

  //Shrink the output size to match maxpooling
  // Friggin functor uses these offsets to compute upper corner x,y in the conv filter computation
  out_rows /= max_pool_height;
  out_cols /= max_pool_width;

  //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
  TensorShape c_shape;
  c_shape.push_back(batch);
  c_shape.push_back(out_rows);
  c_shape.push_back(out_cols);
  c_shape.push_back(out_depth);
  output->resize(c_shape);


  //the strides col and row should be decided
  fused_conv_maxpool_functor<T1, T2, Toutput>(input, batch, input_rows,
          input_cols, in_depth, filter,
          filter_rows, filter_cols, out_depth,
          stride_rows, stride_cols, padding_, output, out_rows,
          out_cols);
}

// For now maxpool strides are ignored and assumed to be on tap boundaries
template <class T1, class T2, class Toutput>
void QuantizedFusedConvMaxPool(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   S_TENSOR mina, S_TENSOR maxa, 
                   S_TENSOR minb, S_TENSOR maxb, 
                   S_TENSOR outmin, S_TENSOR outmax,
                   std::vector<int32_t> strides_, std::vector<int32_t> ksize, Padding padding_) {
  const float min_input = *(mina->read<float>(0, 0));
  const float max_input = *(maxa->read<float>(0, 0));
  const float min_filter = *(minb->read<float>(0, 0));
  const float max_filter = *(maxb->read<float>(0, 0));
  const int32_t offset_input =  FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
  const int32_t offset_filter =  FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
  const int32_t offset_output = 0;
  const int32_t mult_output = 1;
  const int32_t shift_output = 0;

  const int32_t in_depth = input->getShape()[3];
  const int32_t out_depth = filter->getShape()[3];
  const int32_t input_rows = input->getShape()[1];
  const int32_t filter_rows = filter->getShape()[0];
  const int32_t max_pool_width = ksize[1];

  const int32_t input_cols = input->getShape()[2];
  const int32_t filter_cols = filter->getShape()[1];
  const int32_t batch = input->getShape()[0];
  const int32_t max_pool_height = ksize[2];

  const int stride_rows = strides_[0];
  const int stride_cols = strides_[1];

  int32_t out_rows, out_cols;
  if (padding_ == VALID) {
    out_rows = (input_rows - filter_rows) / stride_rows + 1;
    out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    out_rows = input_rows;
    out_cols = input_cols;
  }

  //Shrink the output size to match maxpooling
  // Friggin functor uses these offsets to compute upper corner x,y in the conv filter computation
  out_rows /= max_pool_height;
  out_cols /= max_pool_width;

  // 1D Case makes this zero
  int32_t out_rows_s = (out_rows > 0) ? out_rows : 0;
  int32_t out_cols_s = (out_cols > 0) ? out_cols : 0;


  //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
  TensorShape c_shape;
  c_shape.push_back(batch);
  c_shape.push_back(out_rows_s);
  c_shape.push_back(out_cols_s);
  c_shape.push_back(out_depth);
  output->resize(c_shape);


  //the strides col and row should be decided
  quant_fused_conv_maxpool_functor<T1, T2, Toutput>(input, batch, input_rows,
          input_cols, in_depth, offset_input, filter,
          filter_rows, filter_cols, out_depth,
          offset_filter, stride_rows, stride_cols, padding_, output, out_rows,
          out_cols, shift_output, offset_output, mult_output);
  
  float min_output_value;
  float max_output_value;
  QuantizationRangeForMultiplication<T1, T2, Toutput>(                                                     
          min_input, max_input, min_filter, max_filter, &min_output_value,
          &max_output_value);
  float* c_max = outmax->write<float>(0, 0);
  float* c_min = outmin->write<float>(0, 0);
  *c_max = max_output_value;
  *c_min = min_output_value;                                    
}

template <class T1, class T2, class Toutput>
void Conv(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   std::vector<int32_t> strides_, Padding padding_) {
  const int32_t in_depth = input->getShape()[3];
  const int32_t out_depth = filter->getShape()[3];
  const int32_t input_rows = input->getShape()[1];
  const int32_t filter_rows = filter->getShape()[0];

  const int32_t input_cols = input->getShape()[2];
  const int32_t filter_cols = filter->getShape()[1];
  const int32_t batch = input->getShape()[0];

  const int stride_rows = strides_[1];
  const int stride_cols = strides_[2];

  int32_t out_rows, out_cols;
  if (padding_ == VALID) {
    out_rows = (input_rows - filter_rows) / stride_rows + 1;
    out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else {
    // SAME
    out_rows = input_rows;
    out_cols = input_cols;
  }
  //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
  TensorShape c_shape;
  c_shape.push_back(batch);
  c_shape.push_back(out_rows);
  c_shape.push_back(out_cols);
  c_shape.push_back(out_depth);
  output->resize(c_shape);


  //the strides col and row should be decided
  conv_functor<T1, T2, Toutput>(input, batch, input_rows,
          input_cols, in_depth, filter,
          filter_rows, filter_cols, out_depth,
          stride_rows, stride_cols, padding_, output, out_rows,
          out_cols);
}

template <class T1, class T2, class Toutput>
void QuantizedConv(S_TENSOR input, S_TENSOR filter, S_TENSOR output,
                   S_TENSOR mina, S_TENSOR maxa, 
                   S_TENSOR minb, S_TENSOR maxb, 
                   S_TENSOR outmin, S_TENSOR outmax,
                   std::vector<int32_t> strides_, Padding padding_) {
  const float min_input = *(mina->read<float>(0, 0));
  const float max_input = *(maxa->read<float>(0, 0));
  const float min_filter = *(minb->read<float>(0, 0));
  const float max_filter = *(maxb->read<float>(0, 0));
  const int32_t offset_input =  FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
  const int32_t offset_filter =  FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
  const int32_t offset_output = 0;
  const int32_t mult_output = 1;
  const int32_t shift_output = 0;

  const int64_t in_depth = input->getShape()[3];
  const int64_t out_depth = filter->getShape()[3];
  const int64_t input_rows = input->getShape()[1];
  const int64_t filter_rows = filter->getShape()[0];

  const int64_t input_cols = input->getShape()[2];
  const int64_t filter_cols = filter->getShape()[1];                                 
  const int64_t batch = input->getShape()[0];
                                       
  const int stride_rows = strides_[1];
  const int stride_cols = strides_[2];
  
  int64_t out_rows, out_cols;
  if (padding_ == VALID) {
    out_rows = (input_rows - filter_rows) / stride_rows + 1;
    out_cols = (input_cols - filter_cols) / stride_cols + 1;
  } else { 
    // SAME
    out_rows = input_rows;
    out_cols = input_cols;
  }
  //TensorShape out_shape({batch, out_rows, out_cols, out_depth});
  TensorShape c_shape;
  c_shape.push_back(batch);
  c_shape.push_back(out_rows);
  c_shape.push_back(out_cols);
  c_shape.push_back(out_depth);
  output->resize(c_shape);

  //the strides col and row should be decided
  quant_conv_functor<T1, T2, Toutput>(input, batch, input_rows,
          input_cols, in_depth, offset_input, filter,
          filter_rows, filter_cols, out_depth,
          offset_filter, stride_rows, stride_cols, padding_, output, out_rows, 
          out_cols, shift_output, offset_output, mult_output);
                                       
  float min_output_value;
  float max_output_value;
  QuantizationRangeForMultiplication<T1, T2, Toutput>(                                                     
          min_input, max_input, min_filter, max_filter, &min_output_value,
          &max_output_value);
  float* c_max = outmax->write<float>(0, 0);
  float* c_min = outmin->write<float>(0, 0);
  *c_max = max_output_value;
  *c_min = min_output_value;                                    
}


template<class T1, class T2, class TOut>
class ConvOp : public Operator {
  public:
  ConvOp(std::initializer_list<int32_t> strides, Padding padding) {
    std::vector<int32_t> vec_strides;
    for (auto s : strides) {
      vec_strides.push_back(s);
    }
    _setup(vec_strides, padding);
  }
  ConvOp(std::vector<int32_t>& strides, Padding& padding) {
    _setup(strides, padding);
  }
  virtual void compute() override {
    Conv<T1, T2, TOut>(inputs[0], inputs[1], outputs[0], _strides, _padding);
  }
  private:
  std::vector<int32_t> _strides;
  Padding _padding;
  void _setup(std::vector<int32_t>& strides, Padding& padding){
    _strides = strides;
    _padding = padding;
    n_inputs = 2;
    n_outputs = 1;
  }
};

template<class T1, class T2, class TOut>
class FusedConvMaxpoolOp : public Operator {
  public:
  FusedConvMaxpoolOp(std::initializer_list<int32_t> strides, std::initializer_list<int32_t> ksize, Padding padding) {
    std::vector<int32_t> vec_strides;
    std::vector<int32_t> vec_ksize;

    for (auto s : strides) {
      vec_strides.push_back(s);
    }
    for (auto s : ksize) {
      vec_ksize.push_back(s);
    }
    _setup(vec_strides, vec_ksize, padding);
  }
  FusedConvMaxpoolOp(std::vector<int32_t>& strides, std::vector<int32_t> ksize, Padding& padding) {
    _setup(strides, ksize, padding);
  }
  virtual void compute() override {
    FusedConvMaxPool<T1, T2, TOut>(inputs[0], inputs[1], outputs[0], _strides, _ksize, _padding);
  }
  private:
  std::vector<int32_t> _strides;
  std::vector<int32_t> _ksize;
  Padding _padding;
  void _setup(std::vector<int32_t>& strides, std::vector<int32_t>& ksize, Padding& padding){
    _strides = strides;
    _padding = padding;
    _ksize = ksize;
    n_inputs = 2;
    n_outputs = 1;
  }
};

template<class T1, class T2, class TOut>
class QuantizedFusedConvMaxpoolOp : public Operator {
  public:
  QuantizedFusedConvMaxpoolOp(std::initializer_list<int32_t> strides, std::initializer_list<int32_t> ksize, Padding padding) {
    std::vector<int32_t> vec_strides;
    std::vector<int32_t> vec_ksize;

    for (auto s : strides) {
      vec_strides.push_back(s);
    }
    for (auto s : ksize) {
      vec_ksize.push_back(s);
    }
    _setup(vec_strides, vec_ksize, padding);
  }
  QuantizedFusedConvMaxpoolOp(std::vector<int32_t>& strides, std::vector<int32_t> ksize, Padding& padding) {
    _setup(strides, ksize, padding);
  }
  virtual void compute() override {
    QuantizedFusedConvMaxPool<T1, T2, TOut>(inputs[0], inputs[1], outputs[0], inputs[2],
            inputs[3], inputs[4], inputs[5], outputs[1], outputs[2],
            _strides, _ksize, _padding);
  }
  private:
  std::vector<int32_t> _strides;
  std::vector<int32_t> _ksize;
  Padding _padding;
  void _setup(std::vector<int32_t>& strides, std::vector<int32_t>& ksize, Padding& padding){
    _strides = strides;
    _padding = padding;
    _ksize = ksize;
    n_inputs = 6;
    n_outputs = 3;
  }
};

template <class T1, class T2, class TOut>
class MatMulOp : public Operator {
  public:
  MatMulOp() {
    n_inputs = 2;
    n_outputs = 1;
  }
  virtual void compute() override {
    MatMul2<T1, T2, TOut>(inputs[0], inputs[1],
     outputs[0]);
  }
};


template<class T1, class T2, class TOut>
class QntConvOp : public Operator {
  public:
  QntConvOp(std::initializer_list<int32_t> strides, Padding padding) {
    std::vector<int32_t> vec_strides;
    for (auto s : strides) {
      vec_strides.push_back(s);
    }
    _setup(vec_strides, padding);
  }
  QntConvOp(std::vector<int32_t>& strides, Padding& padding) {
    _setup(strides, padding);
  }
  virtual void compute() override {
    QuantizedConv<T1, T2, TOut>(inputs[0], inputs[1], outputs[0], inputs[2], 
    inputs[3], inputs[4], inputs[5], outputs[1], outputs[2],
    _strides, _padding);
  }
  private:
  std::vector<int32_t> _strides;
  Padding _padding;
  void _setup(std::vector<int32_t>& strides, Padding& padding){
    _strides = strides;
    _padding = padding;
    n_inputs = 6;
    n_outputs = 3;
  }
};
template <class T1, class T2, class TOut>
class QntMatMulOp : public Operator {
  public:
  QntMatMulOp() {
    n_inputs = 6;
    n_outputs = 3;
  }
  virtual void compute() override {
    QuantizedMatMul2<T1, T2, TOut>(inputs[0], inputs[3],
     outputs[0], inputs[1], inputs[4], inputs[2], inputs[5],
      outputs[1], outputs[2]);
  }
};

#endif
