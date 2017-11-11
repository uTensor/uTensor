#ifndef DEEP_MNIST_MLP_HPP
#define DEEP_MNIST_MLP_HPP

#include "mbed.h"
#include "tensor.hpp"
#include "test.hpp"
#include "tensorIdxImporter.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"
#include "NnOps.hpp"
#include "ArrayOps.hpp"
#include "uTensor_util.hpp"

template <class T1, class T2, class T3, class T4>
void tensorQuantize(Tensor* input, Tensor** output,
  Tensor** out_min, Tensor** out_max) {

    //reshape
    Tensor* reshape_shape = new RamTensor<int>({1});
    Tensor* reduce_dim = new RamTensor<int>({1});
    Shape input_shape = input->getShape();
    Tensor* reshape_out = nullptr;

    *(reshape_shape->write<int>(0, 0)) = -1;
    *(reduce_dim->write<int>(0, 0)) = 0;

    reshape<float>(input, reshape_shape, &reshape_out);

    //Min and Max of (reshaped) input
    Tensor* min_out = new RamTensor<float>({1});
    Tensor* max_out = new RamTensor<float>({1});
    Min<float, int, float>(reshape_out, reduce_dim, min_out);
    Max<float, int, float>(reshape_out, reduce_dim, max_out);

    tensorChkAlloc<T2>(output, input->getShape());
    delete input;
    Shape shape_one;
    shape_one.push_back(1);
    tensorChkAlloc<T3>(out_min, shape_one);
    tensorChkAlloc<T4>(out_max, shape_one);

    QuantizeV2<T2>(reshape_out, min_out, max_out, *output, *out_min, *out_max);
}

template<class T1, class T2, class T3, class T4, class T5>
void ReluLayer(Tensor* x, Tensor* x_min, Tensor* x_max,
   Tensor* w, Tensor* w_min, Tensor* w_max, Tensor* b,
    Tensor** output, Tensor** output_min, Tensor** output_max) {
  
    //quantized matmul
    Tensor* out_c = nullptr;
    Tensor* matmul_out_min = new RamTensor<float>({1});
    Tensor* matmul_out_max = new RamTensor<float>({1});

    QuantizedMatMul<uint8_t, uint8_t, int>(x, w, &out_c, x_min, w_min, x_max,
      w_max, matmul_out_min, matmul_out_max);
    //clean up
    delete x;
    delete w;
    delete x_min;
    delete w_min;
    delete x_max;
    delete w_max;

    //Requantization_Range
    Tensor* req_out_min = new RamTensor<float>({1});
    Tensor* req_out_max = new RamTensor<float>({1});
    Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max);

    //Requantize
    Tensor* reqnt_out = new RamTensor<unsigned char>(out_c->getShape());
    Tensor* reqnt_out_min = new RamTensor<float>({1});
    Tensor* reqnt_out_max = new RamTensor<float>({1});
    Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max,
      reqnt_out, reqnt_out_min, reqnt_out_max);

    Shape out_shape = out_c->getShape();
    //clean up
    delete out_c;
    delete matmul_out_min;
    delete matmul_out_max;
    delete req_out_min;
    delete req_out_max;

    Tensor* deqnt_out = nullptr;
    tensorChkAlloc<float>(&deqnt_out, reqnt_out->getShape());
    dequantize<unsigned char>(reqnt_out, reqnt_out_min, reqnt_out_max, &deqnt_out);
    delete reqnt_out;

    Tensor* z_output = new RamTensor<float>(deqnt_out->getShape()); 
    Add<float, float>(deqnt_out, b, &z_output);
    delete deqnt_out;
    delete b;

    Tensor* z_qnt_output = nullptr;
    Tensor* z_min = new RamTensor<float>({1});
    Tensor* z_max = new RamTensor<float>({1});
    tensorQuantize<float, unsigned char, float, float>(z_output, &z_qnt_output, &z_min, &z_max);

    tensorChkAlloc<T3>(output, z_qnt_output->getShape());
    Shape shape_one;
    shape_one.push_back(1);
    tensorChkAlloc<T4>(output_min, shape_one);
    tensorChkAlloc<T4>(output_max, shape_one);
    Relu<unsigned char, float, unsigned char>(z_qnt_output, z_min, z_max, *output, *output_min, *output_max);
}

template<class T1, class T2, class T3>
void PredLayer(Tensor* input, Tensor* input_min,
               Tensor* input_max, Tensor** output, Tensor* w, Tensor* w_min, Tensor* w_max, Tensor* bias, Tensor* dim) {

  Tensor* out_c = nullptr;
  Tensor* matmul_out_min = new RamTensor<float>({1});
  Tensor* matmul_out_max = new RamTensor<float>({1});

  //MatMul
  QuantizedMatMul<uint8_t, uint8_t, int>(input, w, &out_c, input_min, w_min,
                                         input_max, w_max, matmul_out_min,
                                         matmul_out_max);
  //clean up
  delete input;
  delete w;;
  delete w_min;
  delete w_max;
  delete input_min;
  delete input_max;

  //Requantization_Range
  Tensor* req_out_min = new RamTensor<float>({1});
  Tensor* req_out_max = new RamTensor<float>({1});
  Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max,
                                   req_out_min, req_out_max);

  //Requantize
  Tensor* reqnt_out = new RamTensor<unsigned char>(out_c->getShape());
  Tensor* reqnt_out_min = new RamTensor<float>({1});
  Tensor* reqnt_out_max = new RamTensor<float>({1});
  Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max,
                                        req_out_min, req_out_max, reqnt_out,
                                        reqnt_out_min, reqnt_out_max);

  delete out_c;
  delete matmul_out_min;
  delete matmul_out_max;

  //dequantize
  Tensor* deqnt_out = nullptr;
  dequantize<unsigned char>(reqnt_out, reqnt_out_min, reqnt_out_max, &deqnt_out);
  delete reqnt_out_min;
  delete reqnt_out_max;

  //Add
  Tensor* output_z = nullptr;
  Add<float, float>(deqnt_out, bias, &output_z);
  delete deqnt_out;
  delete bias;

  //ArgMax
  ArgMax<float, T3>(output_z, dim, output);
}

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
