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
    Tensor<float> mnist_input = t_import.float_import("/fs/testData/idxImport/float_4d_power2.idx");
    Tensor<int> reshape_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    //output
    Tensor<float> reshape_out;
    reshape(mnist_input, reshape_dim, reshape_out);
    mnist_input.~Tensor();
    reshape_dim.~Tensor();


    //min
    //input
    Tensor<int> min_reduce_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    //output
    Tensor<float> min_out({1});
    Min(reshape_out, min_reduce_dim, min_out);
    min_reduce_dim.~Tensor();

    //max
    //input
    Tensor<int> max_reduce_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
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

    Tensor<unsigned char> qnt_ref = t_import.ubyte_import("/fs/testData/qB/out/qB_0.idx");
    Tensor<float> qnt_min_ref = t_import.float_import("/fs/testData/qB/out/qB_1.idx");
    Tensor<float> qnt_max_ref = t_import.float_import("/fs/testData/qB/out/qb_2.idx");

    double result = meanPercentErr(qnt_ref, qnt_out);
    result += meanPercentErr(qnt_min_ref, qnt_min);
    result += meanPercentErr(qnt_max_ref, qnt_max);

    timer_stop();

    passed(result == 0);
  }

  //quantized matmul dequant add
  //layer value prior to activation function
  void runQntDeqntLayerZ() {

    //quantized matrix multiplication
    //input
    Tensor<unsigned char> a =
      t_import.ubyte_import("/fs/testData/qMatMul/in/qA_0.idx");
    Tensor<float> a_min =
      t_import.float_import("/fs/testData/qMatMul/in/qA_1.idx");
    Tensor<float> a_max =
      t_import.float_import("/fs/testData/qMatMul/in/qA_2.idx");
    Tensor<unsigned char> b =
      t_import.ubyte_import("/fs/testData/qMatMul/in/qB_0.idx");
    Tensor<float> b_min =
      t_import.float_import("/fs/testData/qMatMul/in/qB_1.idx");
    Tensor<float> b_max =
      t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");
    //output
    uint32_t out_row = (a.getShape())[1];
    uint32_t out_col = (b.getShape())[0];
    Tensor<int> out_c({out_row, out_col});

    Tensor<float> matmul_out_min({1});
    Tensor<float> matmul_out_max({1});

    QuantizedMatMul<uint8_t, uint8_t, int>(a, b, out_c, a_min, b_min, a_max,
      b_max, matmul_out_min, matmul_out_max);
    //clean up
    a.~Tensor();
    b.~Tensor();
    a_min.~Tensor();
    b_min.~Tensor();
    a_max.~Tensor();
    b_max.~Tensor();

    //output
    Tensor<float> req_out_min({1});
    Tensor<float> req_out_max({1});
    Requantization_Range<int, float>(out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max);

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

    //output
    Tensor<float> deqnt_out(out_c.getShape());
    dequantize(out_c, reqnt_out_min, reqnt_out_max, deqnt_out);
    out_c.~Tensor();
    reqnt_out_min.~Tensor();
    reqnt_out_max.~Tensor();


    //input
    Tensor<float> bias = t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");
    //output
    Tensor<float> output_z(deqnt_out.getShape()); 
    Add<float, float>(deqnt_out, bias, output_z);

    //load reference
    Tensor<float> ref_z = t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");
    
    double result = meanPercentErr(ref_z, output_z);

    passed(result == 0);

  }

  void runQntRelu() {
    Tensor<float> input_z = t_import.float_import("/fs/testData/qMatMul/in/qB_2.idx");
    Tensor<int> reshape_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");

    Tensor<float> reshape_out;
    reshape(input_z, reshape_dim, reshape_out);

    //min
    //input
    Tensor<int> min_reduce_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
    //output
    Tensor<float> min_out({1});
    Min(reshape_out, min_reduce_dim, min_out);
    min_reduce_dim.~Tensor();

    //max
    //input
    Tensor<int> max_reduce_dim = t_import.int_import("/fs/testData/idxImport/int32_4d_power2.idx");
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
    
  }


  void runAll() {
    runQuantization();
    runQntDeqntLayerZ();
    runQntRelu();
  }
};

#endif  //UTENSOR_MLP_TEST