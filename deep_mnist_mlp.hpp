#ifndef DEEP_MNIST_MLP_HPP
#define DEEP_MNIST_MLP_HPP
#include "mbed.h"
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"
#include "NnOps.hpp"
#include "ArrayOps.hpp"
#include "uTensor_util.hpp"
#include "uTensorTest.hpp" //Need to move math functions to other library

void tensorQuantize(Tensor<float> input, Tensor<unsigned char> &output,
  Tensor<float> &out_min, Tensor<float> &out_max);

void ReluLayer(Tensor<unsigned char> x, Tensor<float> x_min, Tensor<float> x_max,
   Tensor<unsigned char> w, Tensor<float> w_min, Tensor<float> w_max, Tensor<float> b,
    Tensor<unsigned char> &output, Tensor<float> &output_min, Tensor<float> &output_max);

void PredLayer(Tensor<unsigned char> input, Tensor<float> input_min,
               Tensor<float> input_max, Tensor<int> &output);

//Test code
/*
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
*/

int runMLP(string inputIdxFile);

#endif
