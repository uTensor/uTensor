#include "deep_mnist_mlp.hpp"

int runMLP(string inputIdxFile) {
  TensorIdxImporter t_import;
  Tensor* x =
      t_import.float_import(inputIdxFile);
  Tensor* x_quantized = nullptr;
  Tensor* x_min = nullptr;
  Tensor* x_max = nullptr;

  tensorQuantize<float, unsigned char, float, float>(x, &x_quantized, &x_min, &x_max);

  Tensor* w = t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_quint8_const_0.idx");
  Tensor* w_min =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_min_0.idx");
  Tensor* w_max =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_max_0.idx");
  Tensor* b =
      t_import.float_import("/fs/testData/deep_mlp/import-Variable_1_0.idx");
  Tensor* relu_output = nullptr;
  Tensor* relu_min = nullptr;
  Tensor* relu_max = nullptr;

  ReluLayer<unsigned char, unsigned char, unsigned char, float, float>(x_quantized, x_min, x_max, w, w_min, w_max, b, &relu_output,
            &relu_min, &relu_max);

  w = t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_2_quint8_const_0.idx");
  w_min = t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_min_0.idx");
  w_max = t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_max_0.idx");
  b = t_import.float_import("/fs/testData/deep_mlp/import-Variable_3_0.idx");
  Tensor* relu_output2 = nullptr;
  Tensor* relu_min2 = nullptr;
  Tensor* relu_max2 = nullptr;

  ReluLayer<unsigned char, unsigned char, unsigned char, float, float>(relu_output, relu_min, relu_max, w, w_min, w_max, b, &relu_output2,
            &relu_min2, &relu_max2);


  Tensor* pred = nullptr;
  PredLayer<unsigned char, float, int>(relu_output2, relu_min2, relu_max2, &pred);


  Tensor* ref_out = t_import.float_import(
    "/fs/testData/deep_mlp/runPredLayer/y_pred/outputs/y_pred_0.idx");
  Tensor* ref_pred = TensorCast<float, int>(ref_out);

  double result = meanPercentErr<int>(ref_pred, pred);
  
  if (result < 0.0001) {
    printf("PASSED %.8f\r\n\r\n", result);
  } else {
    printf("FAILED %.8f\r\n\r\n", result);
  }

  return *(pred->read<int>(0, 0));
  // output layer
}
