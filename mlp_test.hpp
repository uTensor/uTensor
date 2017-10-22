#ifndef UTENSOR_MLP_TEST
#define UTENSOR_MLP_TEST

#include "tensor.hpp"
#include "ArrayOps.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"

class mlpTest : public Test {
public:
  TensorIdxImporter t_import;

  void runQuantization() {

    testStart("runQuantization");
    timer_start();

    //reshape
    //input
    Tensor<float> mnist_input = t_import.float_import("/fs/testData/mlpTest/runQuantization/in/import-Placeholder_0.idx");
    Tensor<int> reshape_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reshape_dims_0.idx");
    //output
    Tensor<float> reshape_out;
    reshape(mnist_input, reshape_dim, reshape_out);
    mnist_input.~Tensor();
    reshape_dim.~Tensor();


    //min
    //input
    Tensor<int> min_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_min.idx");
    //output
    Tensor<float> min_out({1});
    Min(reshape_out, min_reduce_dim, min_out);
    min_reduce_dim.~Tensor();

    //max
    //input
    Tensor<int> max_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_max.idx");
    //output
    Tensor<float> max_out({1});
    Max(reshape_out, max_reduce_dim, max_out);
    max_reduce_dim.~Tensor();

    //quantization
    //output
    Tensor<unsigned char> qnt_out(reshape_out.getShape());
    Tensor<float> qnt_min({1});
    Tensor<float> qnt_max({1});
    QuantizeV2(reshape_out, min_out, max_out, qnt_out, qnt_min, qnt_max);
    reshape_out.~Tensor();

    timer_stop();

    Tensor<unsigned char> qnt_ref = t_import.ubyte_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_0.idx");
    Tensor<float> qnt_min_ref = t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_1.idx");
    Tensor<float> qnt_max_ref = t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_2.idx");

    double result = meanPercentErr(qnt_ref, qnt_out);
    result += meanPercentErr(qnt_min_ref, qnt_min);
    result += meanPercentErr(qnt_max_ref, qnt_max);

    passed(result == 0);
  }

  //quantized matmul dequant add
  //layer value prior to activation function
  void runQntDeqntLayerZ() {
    DEBUG("running runQntDeqntLayerZ\r\n");
    testStart("runQntDeqntLayerZ");
    timer_start();

    //quantized matrix multiplication
    //input
    Tensor<unsigned char> x =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_0.idx");
    Tensor<float> x_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_1.idx");
    Tensor<float> x_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_2.idx");
    Tensor<unsigned char> w =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_quint8_const_0.idx");
    Tensor<float> w_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_min_0.idx");
    Tensor<float> w_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_max_0.idx");

    DEBUG("all QuantizedMatMul input imported...\r\n");

    //output
    uint32_t out_col = (x.getShape())[0];
    uint32_t out_row = (w.getShape())[1];
    Tensor<int> out_c({out_col, out_row});

    // printf("x[0] = %d, x[1] = %d, b[0] = %d, b[1] = %d\r\n", (x.getShape())[0], (x.getShape())[1],
    // (w.getShape())[0], (w.getShape())[1]);
    // printf("c[0] = %d, c[1] = %d\r\n", (out_c.getShape())[0], (out_c.getShape())[1]);
    // fflush(stdout);

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

    Tensor<int> ref_out_c =
    t_import.int_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_0.idx");
  Tensor<float> ref_matmul_out_min =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_1.idx");
  Tensor<float> ref_matmul_out_max =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_2.idx");

    double temp_result = (meanPercentErr(ref_out_c, out_c) + meanPercentErr(ref_matmul_out_min, matmul_out_min) + meanPercentErr(ref_matmul_out_max, matmul_out_max));
    if(temp_result > 0) {
        printf("matrix mul failed\r\n");
        failed();
        return;
      } else {
        printf("matrix mul passed\r\n");
      }

    DEBUG("QuantizedMatMul completed!\r\n");

    //output
    Tensor<float> req_out_min({1});
    Tensor<float> req_out_max({1});
    Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max);

    Tensor<float> ref_req_out_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requant_range_0.idx");
    Tensor<float> ref_req_out_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requant_range_1.idx");

    temp_result = (meanPercentErr(ref_req_out_min, req_out_min) + meanPercentErr(ref_req_out_max, req_out_max));
      if(temp_result > 0) {
          printf("Requantization_Range failed\r\n");
          failed();
          return;
      } else {
          printf("Requantization_Range passed\r\n");
      }

    DEBUG("Requantization_Range completed!\r\n");

    //output
    Tensor<unsigned char> reqnt_out(out_c.getShape());
    Tensor<float> reqnt_out_min({1});
    Tensor<float> reqnt_out_max({1});
    Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max,
      reqnt_out, reqnt_out_min, reqnt_out_max);
    //clean up
    matmul_out_min.~Tensor();
    matmul_out_max.~Tensor();
    req_out_min.~Tensor();
    req_out_max.~Tensor();

    Tensor<unsigned char> ref_reqnt_out =
    t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_0.idx");
  Tensor<float> ref_reqnt_out_min =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_1.idx");
  Tensor<float> ref_reqnt_out_max =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_2.idx");

    temp_result = (meanPercentErr(ref_reqnt_out, reqnt_out) + meanPercentErr(ref_reqnt_out_min, reqnt_out_min) + meanPercentErr(ref_reqnt_out_max, reqnt_out_max));
    if(temp_result > 0) {
        printf("Requantize failed\r\n");
        failed();
        return;
    } else {
        printf("Requantize passed\r\n");
    }

    DEBUG("Requantize completed!\r\n");

    //output
    Tensor<float> deqnt_out(out_c.getShape());
    dequantize(reqnt_out, reqnt_out_min, reqnt_out_max, deqnt_out);
    out_c.~Tensor();
    reqnt_out_min.~Tensor();
    reqnt_out_max.~Tensor();

    Tensor<float> ref_deqnt_out = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_0.idx");
    double temp;
    if(temp = meanPercentErr(ref_deqnt_out, deqnt_out) > 0) {
      printf("dequantize failed (%.6f)\r\n", temp);
      float* ref_ptr = ref_deqnt_out.getPointer({});
      float* test_ptr = deqnt_out.getPointer({});
      for(uint32_t i; i < ref_deqnt_out.getSize(); i++) {
        if(ref_ptr[i] != test_ptr[i]) {
          printf("%d: %.3f != %.3f, diff: %.8f%%\r\n", i, ref_ptr[i], test_ptr[i], test_ptr[i]/ref_ptr[i]);
        } else {
          printf("%d: %.3f == %.3f\r\n", i, ref_ptr[i], test_ptr[i]);
        }
      }
      failed();
      return;
    } else {
        printf("dequantize passed\r\n");
    }

    DEBUG("dequantize completed!\r\n");

    //input
    Tensor<float> bias = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-Variable_1_0.idx");
    //output
    Tensor<float> output_z(deqnt_out.getShape()); 
    Add<float, float>(deqnt_out, bias, output_z);

    DEBUG("Add completed!\r\n");

    timer_stop();

    //load reference
    Tensor<float> ref_z = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-add_0.idx");

    double result = meanPercentErr(ref_z, output_z);

    passed(result == 0);

  }

  void runQntRelu() {

    testStart("runQntRelu");

    Tensor<float> input_z = t_import.float_import("/fs/testData/mlpTest/runQntRelu/in/import-add_0.idx");
    Tensor<int> reshape_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reshape_dims_0.idx");
    Tensor<float> reshape_out;

    timer_start();

    reshape(input_z, reshape_dim, reshape_out);

    //min
    //input
    Tensor<int> min_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_min.idx");
    //output
    Tensor<float> min_out({1});
    Min(reshape_out, min_reduce_dim, min_out);
    min_reduce_dim.~Tensor();

    //max
    //input
    Tensor<int> max_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_max.idx");
    //output
    Tensor<float> max_out({1});
    Max(reshape_out, max_reduce_dim, max_out);
    max_reduce_dim.~Tensor();

    //quantization
    //output
    Tensor<unsigned char> qnt_out(reshape_out.getShape());
    Tensor<float> qnt_min({1});
    Tensor<float> qnt_max({1});
    QuantizeV2(reshape_out, min_out, max_out, qnt_out, qnt_min, qnt_max);
    reshape_out.~Tensor();
    
    Tensor<unsigned char> out(qnt_out.getShape());
    Tensor<float> out_min({1});
    Tensor<float> out_max({1});
    Relu<unsigned char, float, unsigned char>(qnt_out, qnt_min, qnt_max, out, out_min,
      out_max);

    timer_stop();

    Tensor<unsigned char> ref_out =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_0.idx");
    Tensor<float> ref_out_min =
      t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_1.idx");
    Tensor<float> ref_out_max =
      t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_2.idx");

    double result = meanPercentErr(ref_out, out);
    result += meanPercentErr(ref_out_min, out_min);
    result += meanPercentErr(ref_out_max, out_max);
    

    passed(result == 0);
  }


  void runAll() {
    runQuantization();
    runQntDeqntLayerZ();
    runQntRelu();
  }
};

#endif  //UTENSOR_MLP_TEST