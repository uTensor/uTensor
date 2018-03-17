#ifndef UTENSOR_MLP_TEST
#define UTENSOR_MLP_TEST

#include "tensor.hpp"
#include "ArrayOps.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"
#include "uTensorBase.hpp"
#include "context.hpp"
#include "test.hpp"

class mlpTest : public Test {
public:
  TensorIdxImporter t_import;
  Context ctx;

  void runQuantization() {

    testStart("runQuantization");
    timer_start();

    //reshape
    //input
    S_TENSOR mnist_input = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/in/import-Placeholder_0.idx", "mnist_input"));
    S_TENSOR reshape_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reshape_dims_0.idx", "reshape_dim"));
    //output
    S_TENSOR reshape_out = ctx.add(new RamTensor<float>("reshape_out"));
//    S_TENSOR out_reshape_out = reshape_out.lock();
    ctx.push(new ReshapeOp(), {"mnist_input", "reshape_dim"}, {"reshape_out"});


    //min
    //input
    S_TENSOR min_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_min.idx", "min_reduce_dim"));
    //output
    S_TENSOR min_out = ctx.add(new RamTensor<float>({1}, "min_out"));
 //   S_TENSOR out_min_out = min_out.lock();
    ctx.push(new MinOp(), {"reshape_out", "min_reduce_dim"}, {"min_out"});

    //max
    //input
    S_TENSOR max_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_max.idx", "max_reduce_dim"));
    //output
    S_TENSOR max_out = ctx.add(new RamTensor<float>({1}, "max_out"));
  //  S_TENSOR out_max_out = max_out.lock();
    ctx.push(new MaxOp(), {"reshape_out", "max_reduce_dim"}, {"max_out"});

    //quantization
    //output
    S_TENSOR qnt_out = ctx.add(new RamTensor<unsigned char>("qnt_out"));
    S_TENSOR qnt_min = ctx.add(new RamTensor<float>({1}, "qnt_min"));
    S_TENSOR qnt_max = ctx.add(new RamTensor<float>({1}, "qnt_max"));

    S_TENSOR qnt_ref = ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_0.idx", "qnt_ref"));
    S_TENSOR qnt_min_ref = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_1.idx", "qnt_min_ref"));
    S_TENSOR qnt_max_ref = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_2.idx", "qnt_max_ref"));

    ctx.push(new QuantizeV2Op(), {"reshape_out", "min_out", "max_out"}, {"qnt_out", "qnt_min", "qnt_max"});
    ctx.eval();

    timer_stop();
    double result = meanPercentErr<unsigned char>(qnt_ref.get(), qnt_out.get());
    result += meanPercentErr<float>(qnt_min_ref.get(), qnt_min.get());
    result += meanPercentErr<float>(qnt_max_ref.get(), qnt_max.get());

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
    S_TENSOR x =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_0.idx", "x"));
    S_TENSOR x_min =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_1.idx", "x_min"));
    S_TENSOR x_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_2.idx", "x_max"));
    S_TENSOR w =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_quint8_const_0.idx", "w"));
    S_TENSOR w_min =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_min_0.idx", "w_min"));
    S_TENSOR w_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_max_0.idx", "w_max"));

    DEBUG("all QuantizedMatMul input imported...\r\n");

    //output
    uint32_t out_col = (x->getShape())[0];
    uint32_t out_row = (w->getShape())[1];
    S_TENSOR out_c = ctx.add(new RamTensor<int>({out_col, out_row}, "out_c"));

    // printf("x[0] = %d, x[1] = %d, b[0] = %d, b[1] = %d\r\n", (x.getShape())[0], (x.getShape())[1],
    // (w.getShape())[0], (w.getShape())[1]);
    // printf("c[0] = %d, c[1] = %d\r\n", (out_c.getShape())[0], (out_c.getShape())[1]);
    // fflush(stdout);

    S_TENSOR matmul_out_min = ctx.add(new RamTensor<float>({1}, "matmul_out_min"));
    S_TENSOR matmul_out_max = ctx.add(new RamTensor<float>({1}, "matmul_out_max"));

    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), {"x", "x_min", "x_max", "w", "w_min", "w_max"}, {"out_c", "matmul_out_min", "matmul_out_max"});
    //clean up

   // double temp_result = (meanPercentErr<int>(ref_val.get(), out_val.get()) + meanPercentErr<float>(ref_min.get(), out_min.get()) + meanPercentErr<float>(ref_max.get(), out_max.get()));
    //if(temp_result > 0) {
    //    DEBUG("matrix mul failed\r\n");
    //    failed();
    //    return;
    //  } else {
    //    DEBUG("matrix mul passed\r\n");
    //  }

    DEBUG("QuantizedMatMul completed!\r\n");

    //output
    S_TENSOR req_out_min = ctx.add(new RamTensor<float>({1}, "req_out_min"));
    S_TENSOR req_out_max = ctx.add(new RamTensor<float>({1}, "req_out_max"));
    ctx.push(new Requantization_RangeOp(), {"out_c", "matmul_out_min", "matmul_out_max"}, {"req_out_min", "req_out_max"});


//    temp_result = (meanPercentErr<float>(ref_req_min.get(), out_req_min.get()) + meanPercentErr<float>(ref_req_max.get(), out_req_max.get()));
//      if(temp_result > 0) {
//        DEBUG("Requantization_Range failed\r\n");
//        failed();
//        return;
//      } else {
//        DEBUG("Requantization_Range passed\r\n");
//      }

//    DEBUG("Requantization_Range completed!\r\n");

    //output
    S_TENSOR reqnt_out = ctx.add(new RamTensor<unsigned char>(out_c->getShape(), "reqnt_out"));
    S_TENSOR reqnt_out_min = ctx.add(new RamTensor<float>({1}, "reqnt_out_min"));
    S_TENSOR reqnt_out_max = ctx.add(new RamTensor<float>({1}, "reqnt_out_max"));
    ctx.push(new RequantizeOp(), {"out_c", "matmul_out_min", "matmul_out_max", "req_out_min", "req_out_max"}, {"reqnt_out", "reqnt_out_min", "reqnt_out_max"});
    //clean up

    S_TENSOR ref_reqnt_out =
    ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_0.idx", "ref_reqnt_out"));
  S_TENSOR ref_reqnt_out_min =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_1.idx", "ref_reqnt_out_min"));
  S_TENSOR ref_reqnt_out_max =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_2.idx", "ref_reqnt_out_max"));

//    temp_result = (meanPercentErr<unsigned char>(ref_reqnt.get(), out_reqnt.get()) + meanPercentErr<float>(ref_reqnt_min.get(), out_reqnt_min.get()) + meanPercentErr<float>(ref_reqnt_max.get(), out_reqnt_max.get()));
//    if(temp_result > 0) {
//      DEBUG("Requantize failed\r\n");
//      failed();
//      return;
//    } else {
//      DEBUG("Requantize passed\r\n");
//    }

    DEBUG("Requantize completed!\r\n");

    //output
    S_TENSOR deqnt_out = ctx.add(new RamTensor<float>(out_c->getShape(), "deqnt_out"));
    ctx.push(new DequantizeOp(), {"reqnt_out", "reqnt_out_min", "reqnt_out_max"}, {"deqnt_out"});

    S_TENSOR ref_deqnt_out = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_0.idx", "ref_deqnt_out"));
    //double temp = meanPercentErr<float>(ref_deqnt.get(), out_deqnt.get());
    //if(temp > 0.0001) {
    //  printf("dequantize failed (%.6f)\r\n", temp);
    //  const float* ref_ptr = ref_deqnt.get()->read<float>(0, 0);
    //  const float* test_ptr = out_deqnt.get()->read<float>(0, 0);
    //  for(uint32_t i; i < ref_deqnt->getSize(); i++) {
    //    if(ref_ptr[i] != test_ptr[i]) {
    //      DEBUG("%d: %.3f != %.3f, diff: %.8f%%\r\n", i, ref_ptr[i], test_ptr[i], test_ptr[i]/ref_ptr[i]);
      //  } else {
        //  DEBUG("%d: %.3f == %.3f\r\n", i, ref_ptr[i], test_ptr[i]);
      //  }
     // }
     // failed();
     // return;
   // } else {
   //   DEBUG("dequantize passed\r\n");
   // }

    DEBUG("dequantize completed!\r\n");

    //input
    S_TENSOR bias = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-Variable_1_0.idx", "bias"));
    //output
    S_TENSOR output_z = ctx.add(new RamTensor<float>(deqnt_out->getShape(), "output_z")); 
    ctx.push(new AddOp<float, float>(), {"deqnt_out", "bias"}, {"output_z"});
    ctx.eval();

    DEBUG("Add completed!\r\n");

    timer_stop();

    //load reference
    S_TENSOR ref_z = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-add_0.idx", "ref_z"));

    double result = meanPercentErr<float>(ref_z.get(), output_z.get());

    passed(result < 0.0001);

  }

  void runQntRelu() {

    testStart("runQntRelu");

    S_TENSOR input_z = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/in/import-add_0.idx", "input_z1"));
    S_TENSOR reshape_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reshape_dims_0.idx", "reshape_dim1"));
    S_TENSOR reshape_out = ctx.add(new RamTensor<float>("reshape_out1"));

    timer_start();

    ctx.push(new ReshapeOp(), {"input_z1", "reshape_dim1"}, {"reshape_out1"});

    //min
    //input
    S_TENSOR min_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_min.idx", "min_reduce_dim1"));
    //output
    S_TENSOR min_out = ctx.add(new RamTensor<float>({1}, "min_out1"));
    ctx.push(new MinOp(), {"reshape_out1", "min_reduce_dim1"}, {"min_out1"});

    //max
    //input
    S_TENSOR max_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_max.idx", "max_reduce_dim1"));
    //output
    S_TENSOR max_out = ctx.add(new RamTensor<float>({1}, "max_out1"));
    ctx.push(new MaxOp(), {"reshape_out1", "max_reduce_dim1"}, {"max_out1"});

    //quantization
    //output
    S_TENSOR qnt_out = ctx.add(new RamTensor<unsigned char>("qnt_out1"));
    S_TENSOR qnt_min = ctx.add(new RamTensor<float>({1}, "qnt_min1"));
    S_TENSOR qnt_max = ctx.add(new RamTensor<float>({1}, "qnt_max1"));
    ctx.push(new QuantizeV2Op(), {"reshape_out1", "min_out1", "max_out1"}, {"qnt_out1", "qnt_min1", "qnt_max1"});
    
    S_TENSOR out = ctx.add(new RamTensor<unsigned char>("out1"));
    S_TENSOR out_min = ctx.add(new RamTensor<float>({1}, "out_min1"));
    S_TENSOR out_max = ctx.add(new RamTensor<float>({1}, "out_max1"));

    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), {"qnt_out1", "qnt_min1", "qnt_max1"}, {"out1", "out_min1", "out_max1"});
    ctx.eval();

    timer_stop();

    S_TENSOR ref_out =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_0.idx", "ref_out1"));
    S_TENSOR ref_out_min = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_1.idx", "ref_out_min1"));
    S_TENSOR ref_out_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_2.idx", "ref_out_max1"));

    double result = meanPercentErr<unsigned char>(ref_out.get(), out.get());
    result += meanPercentErr<float>(ref_out_min.get(), out_min.get());
    result += meanPercentErr<float>(ref_out_max.get(), out_max.get());
    

    passed(result == 0);
  }


  void runAll() {
    runQuantization();
    runQntDeqntLayerZ();
    runQntRelu();
  }
};

#endif  //UTENSOR_MLP_TEST
