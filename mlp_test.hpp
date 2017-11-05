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
    Tensor* mnist_input = t_import.float_import("/fs/testData/mlpTest/runQuantization/in/import-Placeholder_0.idx");
    Tensor* reshape_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reshape_dims_0.idx");
    //output
    Tensor* reshape_out = nullptr;
    reshape<float>(mnist_input, reshape_dim, &reshape_out);
    delete mnist_input;
    delete reshape_dim;


    //min
    //input
    Tensor* min_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_min.idx");
    //output
    Tensor* min_out = new RamTensor<float>({1});
    Min<float, int, float>(reshape_out, min_reduce_dim, min_out);
    delete min_reduce_dim;

    //max
    //input
    Tensor* max_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_max.idx");
    //output
    Tensor* max_out = new RamTensor<float>({1});
    Max<float, int, float>(reshape_out, max_reduce_dim, max_out);
    delete max_reduce_dim;

    //quantization
    //output
    Tensor* qnt_out = new RamTensor<unsigned char>(reshape_out->getShape());
    Tensor* qnt_min = new RamTensor<float>({1});
    Tensor* qnt_max = new RamTensor<float>({1});
    QuantizeV2<unsigned char>(reshape_out, min_out, max_out, qnt_out, qnt_min, qnt_max);
    delete reshape_out;

    timer_stop();

    Tensor* qnt_ref = t_import.ubyte_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_0.idx");
    Tensor* qnt_min_ref = t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_1.idx");
    Tensor* qnt_max_ref = t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_2.idx");

    double result = meanPercentErr<unsigned char>(qnt_ref, qnt_out);
    result += meanPercentErr<float>(qnt_min_ref, qnt_min);
    result += meanPercentErr<float>(qnt_max_ref, qnt_max);

    passed(result == 0);
    delete qnt_ref;
    delete qnt_min_ref;
    delete qnt_max_ref;
    delete qnt_out;
    delete qnt_min;
    delete qnt_max;
    delete max_out;
    delete min_out;
  }

  //quantized matmul dequant add
  //layer value prior to activation function
  void runQntDeqntLayerZ() {
    DEBUG("running runQntDeqntLayerZ\r\n");
    testStart("runQntDeqntLayerZ");
    timer_start();

    //quantized matrix multiplication
    //input
    Tensor* x =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_0.idx");
    Tensor* x_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_1.idx");
    Tensor* x_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_2.idx");
    Tensor* w =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_quint8_const_0.idx");
    Tensor* w_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_min_0.idx");
    Tensor* w_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_max_0.idx");

    DEBUG("all QuantizedMatMul input imported...\r\n");

    //output
    uint32_t out_col = (x->getShape())[0];
    uint32_t out_row = (w->getShape())[1];
    Tensor* out_c = new RamTensor<int>({out_col, out_row});

    // printf("x[0] = %d, x[1] = %d, b[0] = %d, b[1] = %d\r\n", (x.getShape())[0], (x.getShape())[1],
    // (w.getShape())[0], (w.getShape())[1]);
    // printf("c[0] = %d, c[1] = %d\r\n", (out_c.getShape())[0], (out_c.getShape())[1]);
    // fflush(stdout);

    Tensor* matmul_out_min = new RamTensor<float>({1});
    Tensor* matmul_out_max = new RamTensor<float>({1});

    QuantizedMatMul<uint8_t, uint8_t, int>(x, w, out_c, x_min, w_min, x_max,
      w_max, matmul_out_min, matmul_out_max);
    //clean up
    delete x;
    delete w;
    delete x_min;
    delete w_min;
    delete x_max;
    delete w_max;

    Tensor* ref_out_c =
    t_import.int_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_0.idx");
  Tensor* ref_matmul_out_min =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_1.idx");
  Tensor* ref_matmul_out_max =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_2.idx");

    double temp_result = (meanPercentErr<int>(ref_out_c, out_c) + meanPercentErr<float>(ref_matmul_out_min, matmul_out_min) + meanPercentErr<float>(ref_matmul_out_max, matmul_out_max));
    if(temp_result > 0) {
        DEBUG("matrix mul failed\r\n");
        failed();
        return;
      } else {
        DEBUG("matrix mul passed\r\n");
      }
    delete ref_out_c;
    delete ref_matmul_out_max;
    delete ref_matmul_out_min;

    DEBUG("QuantizedMatMul completed!\r\n");

    //output
    Tensor* req_out_min = new RamTensor<float>({1});
    Tensor* req_out_max = new RamTensor<float>({1});
    Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max);

    Tensor* ref_req_out_min =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_requant_range_0.idx");
    Tensor* ref_req_out_max =
      t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_requant_range_1.idx");

    temp_result = (meanPercentErr<float>(ref_req_out_min, req_out_min) + meanPercentErr<float>(ref_req_out_max, req_out_max));
      if(temp_result > 0) {
        DEBUG("Requantization_Range failed\r\n");
        failed();
        return;
      } else {
        DEBUG("Requantization_Range passed\r\n");
      }
    delete ref_req_out_min;
    delete ref_req_out_max;

    DEBUG("Requantization_Range completed!\r\n");

    //output
    Tensor* reqnt_out = new RamTensor<unsigned char>(out_c->getShape());
    Tensor* reqnt_out_min = new RamTensor<float>({1});
    Tensor* reqnt_out_max = new RamTensor<float>({1});
    Requantize<int, float, unsigned char>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max,
      reqnt_out, reqnt_out_min, reqnt_out_max);
    //clean up
    delete matmul_out_min;
    delete matmul_out_max;
    delete req_out_min;
    delete req_out_max;

    Tensor* ref_reqnt_out =
    t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_0.idx");
  Tensor* ref_reqnt_out_min =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_1.idx");
  Tensor* ref_reqnt_out_max =
    t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_2.idx");

    temp_result = (meanPercentErr<unsigned char>(ref_reqnt_out, reqnt_out) + meanPercentErr<float>(ref_reqnt_out_min, reqnt_out_min) + meanPercentErr<float>(ref_reqnt_out_max, reqnt_out_max));
    if(temp_result > 0) {
      DEBUG("Requantize failed\r\n");
      failed();
      return;
    } else {
      DEBUG("Requantize passed\r\n");
    }
    delete ref_reqnt_out;
    delete ref_reqnt_out_min;
    delete ref_reqnt_out_max;

    DEBUG("Requantize completed!\r\n");

    //output
    Tensor* deqnt_out = new RamTensor<float>(out_c->getShape());
    dequantize<unsigned char>(reqnt_out, reqnt_out_min, reqnt_out_max, deqnt_out);
    delete out_c;
    delete reqnt_out_min;
    delete reqnt_out_max;
    delete reqnt_out;

    Tensor* ref_deqnt_out = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_0.idx");
    double temp;
    if((temp = meanPercentErr<float>(ref_deqnt_out, deqnt_out)) > 0) {
      printf("dequantize failed (%.6f)\r\n", temp);
      float* ref_ptr = ref_deqnt_out->read<float>(0, 0);
      float* test_ptr = deqnt_out->read<float>(0, 0);
      for(uint32_t i; i < ref_deqnt_out->getSize(); i++) {
        if(ref_ptr[i] != test_ptr[i]) {
          DEBUG("%d: %.3f != %.3f, diff: %.8f%%\r\n", i, ref_ptr[i], test_ptr[i], test_ptr[i]/ref_ptr[i]);
        } else {
          DEBUG("%d: %.3f == %.3f\r\n", i, ref_ptr[i], test_ptr[i]);
        }
      }
      failed();
      return;
    } else {
      DEBUG("dequantize passed\r\n");
    }
    delete ref_deqnt_out;

    DEBUG("dequantize completed!\r\n");

    //input
    Tensor* bias = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-Variable_1_0.idx");
    //output
    Tensor* output_z = new RamTensor<float>(deqnt_out->getShape()); 
    Add<float, float>(deqnt_out, bias, output_z);
    delete deqnt_out;

    DEBUG("Add completed!\r\n");

    timer_stop();

    //load reference
    Tensor* ref_z = t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-add_0.idx");

    double result = meanPercentErr<float>(ref_z, output_z);

    passed(result < 0.0001);
    delete ref_z;
    delete output_z;
    delete bias;

  }

  void runQntRelu() {

    testStart("runQntRelu");

    Tensor* input_z = t_import.float_import("/fs/testData/mlpTest/runQntRelu/in/import-add_0.idx");
    Tensor* reshape_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reshape_dims_0.idx");
    Tensor* reshape_out = nullptr;

    timer_start();

    reshape<float>(input_z, reshape_dim, &reshape_out);

    //min
    //input
    Tensor* min_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_min.idx");
    //output
    Tensor* min_out = new RamTensor<float>({1});
    Min<float, int, float>(reshape_out, min_reduce_dim, min_out);
    delete min_reduce_dim;

    //max
    //input
    Tensor* max_reduce_dim = t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_max.idx");
    //output
    Tensor* max_out = new RamTensor<float>({1});
    Max<float, int, float>(reshape_out, max_reduce_dim, max_out);
    delete max_reduce_dim;

    //quantization
    //output
    Tensor* qnt_out = new RamTensor<unsigned char>(reshape_out->getShape());
    Tensor* qnt_min = new RamTensor<float>({1});
    Tensor* qnt_max = new RamTensor<float>({1});
    QuantizeV2<unsigned char>(reshape_out, min_out, max_out, qnt_out, qnt_min, qnt_max);
    delete reshape_out;
    
    Tensor* out = new RamTensor<unsigned char>(qnt_out->getShape());
    Tensor* out_min = new RamTensor<float>({1});
    Tensor* out_max = new RamTensor<float>({1});
    Relu<unsigned char, float, unsigned char>(qnt_out, qnt_min, qnt_max, out, out_min,
      out_max);

    timer_stop();

    Tensor* ref_out =
      t_import.ubyte_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_0.idx");
    Tensor* ref_out_min =
      t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_1.idx");
    Tensor* ref_out_max =
      t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_2.idx");

    double result = meanPercentErr<unsigned char>(ref_out, out);
    result += meanPercentErr<float>(ref_out_min, out_min);
    result += meanPercentErr<float>(ref_out_max, out_max);
    

    passed(result == 0);
  }


  void runAll() {
    runQuantization();
    runQntDeqntLayerZ();
    runQntRelu();
  }
};

#endif  //UTENSOR_MLP_TEST
