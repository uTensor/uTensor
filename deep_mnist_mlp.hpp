#include "mbed.h"
#include "tensor.hpp"
#include "test.hpp"
#include "uTensor_util.hpp"

template <typename T>
void tensorChkAlloc(Tensor<T> &t, Shape dim) {
  if (t.getSize() == 0) {
    t = Tensor<T>(dim);
  } else if (t.getShape() != dim) {
    ERR_EXIT("Dim mismatched...\r\n");
  }
}

void tensorQuantize(Tensor<float> input, Tensor<unsigned char> &output,
  Tensor<float> &out_min, Tensor<float> &out_max) {

    Tensor<int> reshape_shape({1});
    *(reshape_shape.getPointer({0})) = -1;  ///NT: WTF am I doing?

    Tensor<int> reduce_dim({1});
    *(reduce_dim.getPointer({0})) = 0;

    Shape input_shape = input.getShape();

    Tensor<float> reshape_out;
    reshape(input, reshape_shape, reshape_out);
    input.~Tensor();

    // printf("reshape_out ");
    // printDim(reshape_out);

    Tensor<float> min_out({1});
    Min(reshape_out, reduce_dim, min_out);

    Tensor<float> max_out({1});
    Max(reshape_out, reduce_dim, max_out);

    float *debug_min = min_out.getPointer({0});
    float *debug_max = max_out.getPointer({0});
    // printf("debug min: %.4f\r\n", *debug_min);
    // printf("debug max: %.4f\r\n", *debug_max);

    tensorChkAlloc(output, input_shape);
    Shape shape_one;
    shape_one.push_back(1);
    tensorChkAlloc(out_min, shape_one);
    tensorChkAlloc(out_max, shape_one);

    QuantizeV2(reshape_out, min_out, max_out, output, out_min, out_max);
}

void ReluLayer(Tensor<unsigned char> x, Tensor<float> x_min, Tensor<float> x_max,
   Tensor<unsigned char> w, Tensor<float> w_min, Tensor<float> w_max, Tensor<float> b,
    Tensor<unsigned char> &output, Tensor<float> &output_min, Tensor<float> &output_max) {
  
    uint32_t out_col = (x.getShape())[0];
    uint32_t out_row = (w.getShape())[1];
    //printf("x.shape: %lu, %lu  w.shape: %lu, %lu\r\n", (x.getShape())[0], (x.getShape())[1], (w.getShape())[0], (w.getShape())[1]);
    Tensor<int> out_c({out_col, out_row});

    Tensor<float> matmul_out_min({1});
    Tensor<float> matmul_out_max({1});

    QuantizedMatMul<uint8_t, uint8_t, int>(x, w, out_c, x_min, w_min, x_max,
      w_max, matmul_out_min, matmul_out_max);
    //clean up
    x.~Tensor();
    w.~Tensor();
    x_min.~Tensor();
    w_min.~Tensor();
    x_max.~Tensor();
    w_max.~Tensor();

    Tensor<float> req_out_min({1});
    Tensor<float> req_out_max({1});
    Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max);


    Tensor<unsigned char> reqnt_out(out_c.getShape());
    Tensor<float> reqnt_out_min({1});
    Tensor<float> reqnt_out_max({1});
    Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max,
      reqnt_out, reqnt_out_min, reqnt_out_max);

    Shape out_shape = out_c.getShape();
    //clean up
    out_c.~Tensor();
    matmul_out_min.~Tensor();
    matmul_out_max.~Tensor();
    req_out_min.~Tensor();
    req_out_max.~Tensor();

    Tensor<float> deqnt_out;
    tensorChkAlloc(deqnt_out, reqnt_out.getShape());
    dequantize(reqnt_out, reqnt_out_min, reqnt_out_max, deqnt_out);
    reqnt_out.~Tensor();

    Tensor<float> z_output(deqnt_out.getShape()); 
    Add<float, float>(deqnt_out, b, z_output);
    deqnt_out.~Tensor();

    Tensor<unsigned char> z_qnt_output;
    Tensor<float> z_min({1});
    Tensor<float> z_max({1});
    tensorQuantize(z_output, z_qnt_output, z_min, z_max);
    z_output.~Tensor();

    tensorChkAlloc(output, z_qnt_output.getShape());
    Shape shape_one;
    shape_one.push_back(1);
    tensorChkAlloc(output_min, shape_one);
    tensorChkAlloc(output_max, shape_one);
    Relu<unsigned char, float, unsigned char>(z_qnt_output, z_min, z_max, output, output_min, output_max);
}

void PredLayer(Tensor<unsigned char> input, Tensor<float> input_min,
               Tensor<float> input_max, Tensor<int> output) {
  TensorIdxImporter t_import;
  Tensor<unsigned char> w = t_import.ubyte_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_quint8_const_0.idx");
  Tensor<float> w_min = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_min_0.idx");
  Tensor<float> w_max = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_max_0.idx");
  uint32_t out_col = (input.getShape())[0];
  uint32_t out_row = (w.getShape())[1];

  Tensor<int> out_c({out_col, out_row});
  Tensor<float> matmul_out_min({1});
  Tensor<float> matmul_out_max({1});

  QuantizedMatMul<uint8_t, uint8_t, int>(input, w, out_c, input_min, w_min,
                                         input_max, w_max, matmul_out_min,
                                         matmul_out_max);
  Tensor<int> ref_out_c = t_import.int_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "outputs/MatMul_2_eightbit_quantized_mat_mul_0.idx");
  Tensor<float> ref_matmul_out_min = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "outputs/MatMul_2_eightbit_quantized_mat_mul_1.idx");
  Tensor<float> ref_matmul_out_max = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "outputs/MatMul_2_eightbit_quantized_mat_mul_2.idx");

  double temp_result =
      (Test::meanPercentErr(ref_out_c, out_c) +
       Test::meanPercentErr(ref_matmul_out_min, matmul_out_min) +
       Test::meanPercentErr(ref_matmul_out_max, matmul_out_max));
  if (temp_result > 0) {
    DEBUG("matrix mul failed\r\n");
    return;
  } else {
    DEBUG("matrix mul passed\r\n");
  }

  DEBUG("QuantizedMatMul completed!\r\n");

  // output
  input.~Tensor();
  w.~Tensor();
  w_min.~Tensor();
  w_max.~Tensor();
  input_min.~Tensor();
  input_max.~Tensor();
  ref_out_c.~Tensor();
  ref_matmul_out_min.~Tensor();
  ref_matmul_out_max.~Tensor();

  Tensor<float> req_out_min({1});
  Tensor<float> req_out_max({1});
  Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max,
                                   req_out_min, req_out_max);
  Tensor<float> ref_req_out_min = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_requant_range/"
      "outputs/MatMul_2_eightbit_requant_range_0.idx");
  Tensor<float> ref_req_out_max = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_requant_range/"
      "outputs/MatMul_2_eightbit_requant_range_1.idx");

  temp_result = (Test::meanPercentErr(ref_req_out_min, req_out_min) +
                 Test::meanPercentErr(ref_req_out_max, req_out_max));
  if (temp_result > 0) {
    DEBUG("Requantization_Range failed\r\n");
    return;
  } else {
    DEBUG("Requantization_Range passed\r\n");
  }

  DEBUG("Requantization_Range completed!\r\n");

  ref_req_out_min.~Tensor();
  ref_req_out_max.~Tensor();

  Tensor<unsigned char> reqnt_out(out_c.getShape());
  Tensor<float> reqnt_out_min({1});
  Tensor<float> reqnt_out_max({1});
  Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max,
                                        req_out_min, req_out_max, reqnt_out,
                                        reqnt_out_min, reqnt_out_max);

  matmul_out_min.~Tensor();
  matmul_out_max.~Tensor();
  Tensor<unsigned char> ref_reqnt_out = t_import.ubyte_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_requantize/outputs/"
      "MatMul_2_eightbit_requantize_0.idx");
  Tensor<float> ref_reqnt_out_min = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_requantize/outputs/"
      "MatMul_2_eightbit_requantize_1.idx");
  Tensor<float> ref_reqnt_out_max = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_requantize/outputs/"
      "MatMul_2_eightbit_requantize_2.idx");

  temp_result = (Test::meanPercentErr(ref_reqnt_out, reqnt_out) +
                 Test::meanPercentErr(ref_reqnt_out_min, reqnt_out_min) +
                 Test::meanPercentErr(ref_reqnt_out_max, reqnt_out_max));
  if (temp_result > 0) {
    DEBUG("Requantize failed\r\n");
    return;
  } else {
    DEBUG("Requantize passed\r\n");
  }
  ref_reqnt_out.~Tensor();
  req_out_max.~Tensor();
  req_out_max.~Tensor();
  ref_reqnt_out_min.~Tensor();
  ref_reqnt_out_max.~Tensor();

  DEBUG("Requantize completed!\r\n");

  Tensor<float> deqnt_out(out_c.getShape());
  dequantize(reqnt_out, reqnt_out_min, reqnt_out_max, deqnt_out);
  out_c.~Tensor();
  reqnt_out_min.~Tensor();
  reqnt_out_max.~Tensor();
  Tensor<float> ref_deqnt_out = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2/outputs/MatMul_2_0.idx");
  temp_result = Test::meanPercentErr(ref_deqnt_out, deqnt_out);
  if (temp_result > 0.0001) {
    printf("dequantize failed (%.6f)\r\n", temp_result);
    float *ref_ptr = ref_deqnt_out.getPointer({});
    float *test_ptr = deqnt_out.getPointer({});
    for (uint32_t i; i < ref_deqnt_out.getSize(); i++) {
      if (ref_ptr[i] != test_ptr[i]) {
        DEBUG("%d: %.3f != %.3f, diff: %.8f%%\r\n", i, ref_ptr[i], test_ptr[i],
              test_ptr[i] / ref_ptr[i]);
      } else {
        DEBUG("%d: %.3f == %.3f\r\n", i, ref_ptr[i], test_ptr[i]);
      }
    }
    return;
  } else {
    DEBUG("dequantize passed\r\n");
  }

  DEBUG("dequantize completed!\r\n");

  Tensor<float> bias = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/add_2/inputs/Variable_5_0.idx");
  Tensor<float> output_z(deqnt_out.getShape());
  Add<float, float>(deqnt_out, bias, output_z);

  Tensor<int> dim = t_import.int_import(
      "/fs/testData/deep_mlp/runPredLayer/y_pred/inputs/"
      "y_pred-dimension_0.idx");
  ArgMax(output_z, dim, output);
}
void runPred(void) {
  TensorIdxImporter t_import;
  Tensor<unsigned char> x = t_import.ubyte_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Relu_1_eightbit_quantized_0.idx");
  Tensor<float> x_min = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Relu_1_eightbit_quantized_1.idx");
  Tensor<float> x_max = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Relu_1_eightbit_quantized_2.idx");
  Tensor<float> ref_out = t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/y_pred/outputs/y_pred_0.idx");
  Tensor<int> out(ref_out.getShape());

  PredLayer(x, x_min, x_max, out);
  Tensor<float> out_float = TensorCast<int, float>(out);
  double result = Test::meanPercentErr(ref_out, out_float);
  if (result < 0.0001) {
    printf("PASSED %.8f\r\n\r\n", result);
  } else {
    printf("FAILED %.8f\r\n\r\n", result);
  }
}

void runMLP(void) {
  TensorIdxImporter t_import;
  Tensor<float> x =
      t_import.float_import("/fs/testData/deep_mlp/import-Placeholder_0.idx");
  Tensor<unsigned char> x_quantized;
  Tensor<float> x_min;
  Tensor<float> x_max;

//   printf("x ");
//   printDim(x);

  tensorQuantize(x, x_quantized, x_min, x_max);

  Tensor<unsigned char> w = t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_quint8_const_0.idx");
  Tensor<float> w_min =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_min_0.idx");
  Tensor<float> w_max =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_max_0.idx");
  Tensor<float> b =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_1_0.idx");
  Tensor<unsigned char> relu_output;
  Tensor<float> relu_min;
  Tensor<float> relu_max;

  ReluLayer(x_quantized, x_min, x_max, w, w_min, w_max, b, relu_output,
            relu_min, relu_max);

  w = t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_2_quint8_const_0.idx");
  w_min = t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_min_0.idx");
  w_max = t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_max_0.idx");
  b = t_import.float_import("/fs/testData/deep_mlp/import-Variable_3_0.idx");
  Tensor<unsigned char> relu_output2;
  Tensor<float> relu_min2;
  Tensor<float> relu_max2;

  ReluLayer(relu_output, relu_min, relu_max, w, w_min, w_max, b, relu_output2,
            relu_min2, relu_max2);
  w.~Tensor();

  Tensor<unsigned char> ref_relu_output2 = t_import.ubyte_import(
      "/fs/testData/deep_mlp/out/import-Relu_1_eightbit_quantized_0.idx");
  Tensor<float> ref_relu_min2 = t_import.float_import(
      "/fs/testData/deep_mlp/out/import-Relu_1_eightbit_quantized_1.idx");
  Tensor<float> ref_relu_max2 = t_import.float_import(
      "/fs/testData/deep_mlp/out/import-Relu_1_eightbit_quantized_2.idx");

  double result = Test::meanPercentErr(ref_relu_output2, relu_output2);
  result += Test::meanPercentErr(ref_relu_min2, relu_min2);
  result += Test::meanPercentErr(ref_relu_max2, relu_max2);

  if(result < 0.0001) {
    printf("PASSED %.8f\r\n\r\n", result);
  } else {
    printf("FAILED %.8f\r\n\r\n", result);
  }

  // output layer
}
