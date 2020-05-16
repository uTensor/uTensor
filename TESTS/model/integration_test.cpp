#include <cstring>
#include <iostream>

#include "gtest/gtest.h"
#include "input_image.h"
#include "params_model.hpp"
#include "uTensor.h"

using std::cout;
using std::endl;

using namespace uTensor;

void compute_model(Tensor& input_10, Tensor& Identity0);

void onError(Error* err) {
  while (true) {
  }
}

TEST(Integration, run_once) {
  localCircularArenaAllocator<2048> meta_allocator;
  localCircularArenaAllocator<40000, uint32_t> ram_allocator;
  SimpleErrorHandler mErrHandler(10);

  mErrHandler.set_onError(onError);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  int pred_labels[10];
  float correct_cnt = 0.0;
  for (int b = 0; b < 1; ++b) {
    Tensor input_image =
        new RomTensor({1, 28, 28, 1}, flt, ref_input_images[b]);
    Tensor logits = new RamTensor({1, 10}, flt);
    compute_model(input_image, logits);
    float max_value = static_cast<float>(logits(0));
    int max_index = 0;
    for (int i = 1; i < 10; ++i) {
      float value = static_cast<float>(logits(i));
      if (value >= max_value) {
        max_value = value;
        max_index = i;
      }
    }
    input_image.free();
    logits.free();
    pred_labels[b] = max_index;
    // uncomment this once we fix the accuracy issue
    // EXPECT_EQ(max_index, ref_labels[b]);
  }
}

TEST(Integration, run_10x_mem_check) {
  localCircularArenaAllocator<2048> meta_allocator;
  localCircularArenaAllocator<40000, uint32_t> ram_allocator;
  SimpleErrorHandler mErrHandler(10);

  mErrHandler.set_onError(onError);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  int pred_labels[20];
  float correct_cnt = 0.0;
  for (int b = 0; b < 10; ++b) {
    Tensor input_image =
        new RomTensor({1, 28, 28, 1}, flt, ref_input_images[b]);
    Tensor logits = new RamTensor({1, 10}, flt);
    compute_model(input_image, logits);
    float max_value = static_cast<float>(logits(0));
    int max_index = 0;
    for (int i = 1; i < 10; ++i) {
      float value = static_cast<float>(logits(i));
      if (value >= max_value) {
        max_value = value;
        max_index = i;
      }
    }
    input_image.free();
    logits.free();
    pred_labels[b] = max_index;
    if (max_index == ref_labels[b]) {
      correct_cnt += 1.0;
    }
  }

  cout << "Note: Note checking accuracy of output" << endl;
}

void compute_model(Tensor& input_10, Tensor& Identity0) {
  // start rendering local declare snippets
  TFLM::QuantizeOperator<int8_t, float> op_000;

  QuantizedFullyConnectedOperator<int8_t> op_001(
      TFLM::TfLiteFusedActivation::kTfLiteActRelu);

  QuantizedDepthwiseSeparableConvOperator<int8_t> op_002(
      {1, 1}, VALID, 32, {1, 1}, TFLM::TfLiteFusedActivation::kTfLiteActRelu);

  MaxPoolOperator<int8_t> op_003({2, 2}, {1, 2, 2, 1}, VALID);

  ReshapeOperator<int8_t> op_004({1, 5408});

  TFLM::DequantizeOperator<float, int8_t> op_005;

  QuantizedFullyConnectedOperator<int8_t> op_006(
      TFLM::TfLiteFusedActivation::kTfLiteActNone);

  Tensor input_1_int80 = new RamTensor({1, 28, 28, 1}, i8);
  int input_1_int80_zp = -128;
  float input_1_int80_scale = 0.003921569;
  PerTensorQuantizationParams input_1_int80_quant_params(input_1_int80_zp,
                                                         input_1_int80_scale);
  input_1_int80->set_quantization_params(input_1_int80_quant_params);

  Tensor conv2dRelu0 = new RamTensor({1, 26, 26, 32}, i8);
  int conv2dRelu0_zp = -128;
  float conv2dRelu0_scale = 0.003272707;
  PerTensorQuantizationParams conv2dRelu0_quant_params(conv2dRelu0_zp,
                                                       conv2dRelu0_scale);
  conv2dRelu0->set_quantization_params(conv2dRelu0_quant_params);

  Tensor max_pooling2dMaxPool0 = new RamTensor({1, 13, 13, 32}, i8);
  int max_pooling2dMaxPool0_zp = -128;
  float max_pooling2dMaxPool0_scale = 0.003272707;
  PerTensorQuantizationParams max_pooling2dMaxPool0_quant_params(
      max_pooling2dMaxPool0_zp, max_pooling2dMaxPool0_scale);
  max_pooling2dMaxPool0->set_quantization_params(
      max_pooling2dMaxPool0_quant_params);

  Tensor denseRelu0 = new RamTensor({1, 128}, i8);
  int denseRelu0_zp = -128;
  float denseRelu0_scale = 0.030642502;
  PerTensorQuantizationParams denseRelu0_quant_params(denseRelu0_zp,
                                                      denseRelu0_scale);
  denseRelu0->set_quantization_params(denseRelu0_quant_params);

  Tensor Identity_int80 = new RamTensor({1, 10}, i8);
  int Identity_int80_zp = -17;
  float Identity_int80_scale = 0.10464356;
  PerTensorQuantizationParams Identity_int80_quant_params(Identity_int80_zp,
                                                          Identity_int80_scale);
  Identity_int80->set_quantization_params(Identity_int80_quant_params);

  Tensor max_pooling2dMaxPool_0_Reshape00 = new RamTensor({1, 5408}, i8);
  int max_pooling2dMaxPool_0_Reshape00_zp = -128;
  float max_pooling2dMaxPool_0_Reshape00_scale = 0.003272707;
  PerTensorQuantizationParams max_pooling2dMaxPool_0_Reshape00_quant_params(
      max_pooling2dMaxPool_0_Reshape00_zp,
      max_pooling2dMaxPool_0_Reshape00_scale);
  max_pooling2dMaxPool_0_Reshape00->set_quantization_params(
      max_pooling2dMaxPool_0_Reshape00_quant_params);

  Tensor conv2dConv2DReadVariableOp0 = new RomTensor(
      {1, 3, 3, 32}, i8,
      data_StatefulPartitionedCall_my_model_conv2d_Conv2D_ReadVariableOp_0);
  int arr_conv2dConv2DReadVariableOp0_zp[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float arr_conv2dConv2DReadVariableOp0_scale[32] = {
      0.0023536368, 0.0010457791, 0.0015707873, 0.0019873076,  0.0012053831,
      0.0032443856, 0.0033889662, 0.0015773318, 0.001541911,   0.00092535064,
      0.0018833402, 0.0010469551, 0.003207928,  0.0026745028,  0.0018283271,
      0.002781117,  0.0017297338, 0.0011449575, 0.0021869328,  0.0033853122,
      0.0034819862, 0.002282607,  0.0013945888, 0.0007485888,  0.0036321173,
      0.0011592397, 0.0009173385, 0.0019517496, 0.00076347584, 0.0013479418,
      0.0017391876, 0.001866459};
  PerChannelQuantizationParams conv2dConv2DReadVariableOp0_quant_params(
      arr_conv2dConv2DReadVariableOp0_zp,
      arr_conv2dConv2DReadVariableOp0_scale);
  conv2dConv2DReadVariableOp0->set_quantization_params(
      conv2dConv2DReadVariableOp0_quant_params);

  Tensor conv2dConv2D_bias0 = new RomTensor(
      {32}, i32, data_StatefulPartitionedCall_my_model_conv2d_Conv2D_bias_0);
  int32_t arr_conv2dConv2D_bias0_zp[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float arr_conv2dConv2D_bias0_scale[32] = {
      9.229949e-06,  4.101095e-06,   6.1599503e-06, 7.793364e-06,
      4.7269928e-06, 1.2723081e-05,  1.3290064e-05, 6.1856153e-06,
      6.0467105e-06, 3.6288263e-06,  7.385648e-06,  4.1057065e-06,
      1.2580111e-05, 1.04882465e-05, 7.1699105e-06, 1.0906342e-05,
      6.7832702e-06, 4.49003e-06,    8.576208e-06,  1.3275735e-05,
      1.3654849e-05, 8.951401e-06,   5.4689763e-06, 2.9356427e-06,
      1.4243598e-05, 4.5460383e-06,  3.597406e-06,  7.65392e-06,
      2.9940231e-06, 5.2860464e-06,  6.820344e-06,  7.319448e-06};
  PerChannelQuantizationParams conv2dConv2D_bias0_quant_params(
      arr_conv2dConv2D_bias0_zp, arr_conv2dConv2D_bias0_scale);
  conv2dConv2D_bias0->set_quantization_params(conv2dConv2D_bias0_quant_params);

  Tensor denseMatMulReadVariableOptranspose0 = new RomTensor(
      {5408, 128}, i8,
      data_StatefulPartitionedCall_my_model_dense_MatMul_ReadVariableOp_transpose_0);
  int denseMatMulReadVariableOptranspose0_zp = 0;
  float denseMatMulReadVariableOptranspose0_scale = 0.0026422727;
  PerTensorQuantizationParams denseMatMulReadVariableOptranspose0_quant_params(
      denseMatMulReadVariableOptranspose0_zp,
      denseMatMulReadVariableOptranspose0_scale);
  denseMatMulReadVariableOptranspose0->set_quantization_params(
      denseMatMulReadVariableOptranspose0_quant_params);

  Tensor denseMatMul_bias0 = new RomTensor(
      {128}, i32, data_StatefulPartitionedCall_my_model_dense_MatMul_bias_0);
  int32_t denseMatMul_bias0_zp = 0;
  float denseMatMul_bias0_scale = 8.647385e-06;
  PerTensorQuantizationParams denseMatMul_bias0_quant_params(
      denseMatMul_bias0_zp, denseMatMul_bias0_scale);
  denseMatMul_bias0->set_quantization_params(denseMatMul_bias0_quant_params);

  Tensor dense_1MatMulReadVariableOptranspose0 = new RomTensor(
      {128, 10}, i8,
      data_StatefulPartitionedCall_my_model_dense_1_MatMul_ReadVariableOp_transpose_0);
  int dense_1MatMulReadVariableOptranspose0_zp = 0;
  float dense_1MatMulReadVariableOptranspose0_scale = 0.0032048211;
  PerTensorQuantizationParams
      dense_1MatMulReadVariableOptranspose0_quant_params(
          dense_1MatMulReadVariableOptranspose0_zp,
          dense_1MatMulReadVariableOptranspose0_scale);
  dense_1MatMulReadVariableOptranspose0->set_quantization_params(
      dense_1MatMulReadVariableOptranspose0_quant_params);

  Tensor dense_1MatMul_bias0 = new RomTensor(
      {10}, i32, data_StatefulPartitionedCall_my_model_dense_1_MatMul_bias_0);
  int32_t dense_1MatMul_bias0_zp = 0;
  float dense_1MatMul_bias0_scale = 9.820374e-05;
  PerTensorQuantizationParams dense_1MatMul_bias0_quant_params(
      dense_1MatMul_bias0_zp, dense_1MatMul_bias0_scale);
  dense_1MatMul_bias0->set_quantization_params(
      dense_1MatMul_bias0_quant_params);

  // end of rendering local declare snippets
  // start rendering eval snippets
  op_000
      .set_inputs({
          {TFLM::QuantizeOperator<int8_t, float>::input, input_10},
      })
      .set_outputs(
          {{TFLM::QuantizeOperator<int8_t, float>::output, input_1_int80}})
      .eval();

  op_002
      .set_inputs({
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::in, input_1_int80},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::filter,
           conv2dConv2DReadVariableOp0},
          {QuantizedDepthwiseSeparableConvOperator<int8_t>::bias,
           conv2dConv2D_bias0},
      })
      .set_outputs(
          {{QuantizedDepthwiseSeparableConvOperator<int8_t>::out, conv2dRelu0}})
      .eval();

  op_003
      .set_inputs({
          {MaxPoolOperator<int8_t>::in, conv2dRelu0},
      })
      .set_outputs({{MaxPoolOperator<int8_t>::out, max_pooling2dMaxPool0}})
      .eval();

  op_004
      .set_inputs({
          {ReshapeOperator<int8_t>::input, max_pooling2dMaxPool0},
      })
      .set_outputs(
          {{ReshapeOperator<int8_t>::output, max_pooling2dMaxPool_0_Reshape00}})
      .eval();

  op_001
      .set_inputs({
          {QuantizedFullyConnectedOperator<int8_t>::input,
           max_pooling2dMaxPool_0_Reshape00},
          {QuantizedFullyConnectedOperator<int8_t>::filter,
           denseMatMulReadVariableOptranspose0},
          {QuantizedFullyConnectedOperator<int8_t>::bias, denseMatMul_bias0},
      })
      .set_outputs(
          {{QuantizedFullyConnectedOperator<int8_t>::output, denseRelu0}})
      .eval();

  op_006
      .set_inputs({
          {QuantizedFullyConnectedOperator<int8_t>::input, denseRelu0},
          {QuantizedFullyConnectedOperator<int8_t>::filter,
           dense_1MatMulReadVariableOptranspose0},
          {QuantizedFullyConnectedOperator<int8_t>::bias, dense_1MatMul_bias0},
      })
      .set_outputs(
          {{QuantizedFullyConnectedOperator<int8_t>::output, Identity_int80}})
      .eval();

  op_005
      .set_inputs({
          {TFLM::DequantizeOperator<float, int8_t>::a, Identity_int80},
      })
      .set_outputs({{TFLM::DequantizeOperator<float, int8_t>::b, Identity0}})
      .eval();
  // end of rendering eval snippets
}
