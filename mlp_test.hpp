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
    TENSOR mnist_input = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/in/import-Placeholder_0.idx"));
    TENSOR reshape_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reshape_dims_0.idx"));
    //output
    TENSOR reshape_out = ctx.add(new RamTensor<float>());
//    S_TENSOR out_reshape_out = reshape_out.lock();
    ctx.push(new ReshapeOp(), {mnist_input, reshape_dim}, {reshape_out});


    //min
    //input
    TENSOR min_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_min.idx"));
    //output
    TENSOR min_out = ctx.add(new RamTensor<float>({1}));
 //   S_TENSOR out_min_out = min_out.lock();
    ctx.push(new MinOp(), {reshape_out, min_reduce_dim}, {min_out});

    //max
    //input
    TENSOR max_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQuantization/in/import-MatMul_eightbit_reduction_dims_0_max.idx"));
    //output
    TENSOR max_out = ctx.add(new RamTensor<float>({1}));
  //  S_TENSOR out_max_out = max_out.lock();
    ctx.push(new MaxOp(), {reshape_out, max_reduce_dim}, {max_out});

    //quantization
    //output
    TENSOR qnt_out = ctx.add(new RamTensor<unsigned char>());
    TENSOR qnt_min = ctx.add(new RamTensor<float>({1}));
    TENSOR qnt_max = ctx.add(new RamTensor<float>({1}));

    S_TENSOR out_qnt = qnt_out.lock();
    S_TENSOR out_min = qnt_min.lock();
    S_TENSOR out_max = qnt_max.lock();

    TENSOR qnt_ref = ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_0.idx"));
    TENSOR qnt_min_ref = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_1.idx"));
    TENSOR qnt_max_ref = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQuantization/out/import-MatMul_eightbit_quantize_Placeholder_2.idx"));
    S_TENSOR ref_qnt = qnt_ref.lock();
    S_TENSOR ref_max = qnt_max_ref.lock();
    S_TENSOR ref_min = qnt_min_ref.lock();

    ctx.push(new QuantizeV2Op(), {reshape_out, min_out, max_out}, {qnt_out, qnt_min, qnt_max});
    ctx.eval();

    timer_stop();
    double result = meanPercentErr<unsigned char>(ref_qnt.get(), out_qnt.get());
    result += meanPercentErr<float>(ref_min.get(), out_min.get());
    result += meanPercentErr<float>(ref_max.get(), out_max.get());

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
    TENSOR x =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_0.idx"));
    TENSOR x_min =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_1.idx"));
    TENSOR x_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_quantize_Placeholder_2.idx"));
    TENSOR w =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_quint8_const_0.idx"));
    TENSOR w_min =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_min_0.idx"));
    TENSOR w_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-Variable_max_0.idx"));

    DEBUG("all QuantizedMatMul input imported...\r\n");

    //output
    uint32_t out_col = (x.lock()->getShape())[0];
    uint32_t out_row = (w.lock()->getShape())[1];
    TENSOR out_c = ctx.add(new RamTensor<int>({out_col, out_row}));

    // printf("x[0] = %d, x[1] = %d, b[0] = %d, b[1] = %d\r\n", (x.getShape())[0], (x.getShape())[1],
    // (w.getShape())[0], (w.getShape())[1]);
    // printf("c[0] = %d, c[1] = %d\r\n", (out_c.getShape())[0], (out_c.getShape())[1]);
    // fflush(stdout);

    TENSOR matmul_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR matmul_out_max = ctx.add(new RamTensor<float>({1}));

    TList inputs = {x, x_min, x_max, w, w_min, w_max};
    TList outputs = {out_c, matmul_out_min, matmul_out_max};
    S_TENSOR out_val = out_c.lock();
    S_TENSOR out_min = matmul_out_min.lock();
    S_TENSOR out_max = matmul_out_max.lock();
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), inputs, outputs);
    //clean up

    TENSOR ref_out_c =
    ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_0.idx"));
  TENSOR ref_matmul_out_min =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_1.idx"));
  TENSOR ref_matmul_out_max =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_quantized_mat_mul_2.idx"));

  S_TENSOR ref_val = ref_out_c.lock();
  S_TENSOR ref_min = ref_matmul_out_min.lock();
  S_TENSOR ref_max = ref_matmul_out_max.lock();
   /* double temp_result = (meanPercentErr<int>(ref_val.get(), out_val.get()) + meanPercentErr<float>(ref_min.get(), out_min.get()) + meanPercentErr<float>(ref_max.get(), out_max.get()));
    if(temp_result > 0) {
        DEBUG("matrix mul failed\r\n");
        failed();
        return;
      } else {
        DEBUG("matrix mul passed\r\n");
      }
*/
    DEBUG("QuantizedMatMul completed!\r\n");

    //output
    TENSOR req_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR req_out_max = ctx.add(new RamTensor<float>({1}));
    S_TENSOR out_req_min = req_out_min.lock();
    S_TENSOR out_req_max = req_out_max.lock();
    ctx.push(new Requantization_RangeOp(), {out_c, matmul_out_min, matmul_out_max}, {req_out_min, req_out_max});

    TENSOR ref_req_out_min =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_requant_range_0.idx"));
    TENSOR ref_req_out_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/in/import-MatMul_eightbit_requant_range_1.idx"));
    S_TENSOR ref_req_min = ref_req_out_min.lock();
   S_TENSOR ref_req_max = ref_req_out_max.lock(); 
/*
    temp_result = (meanPercentErr<float>(ref_req_min.get(), out_req_min.get()) + meanPercentErr<float>(ref_req_max.get(), out_req_max.get()));
      if(temp_result > 0) {
        DEBUG("Requantization_Range failed\r\n");
        failed();
        return;
      } else {
        DEBUG("Requantization_Range passed\r\n");
      }

    DEBUG("Requantization_Range completed!\r\n");*/

    //output
    TENSOR reqnt_out = ctx.add(new RamTensor<unsigned char>(out_c.lock()->getShape()));
    TENSOR reqnt_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR reqnt_out_max = ctx.add(new RamTensor<float>({1}));
    S_TENSOR out_reqnt = reqnt_out.lock();
    S_TENSOR out_reqnt_min = reqnt_out_min.lock();
    S_TENSOR out_reqnt_max = reqnt_out_max.lock();
    ctx.push(new RequantizeOp(), {out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max}, {reqnt_out, reqnt_out_min, reqnt_out_max});
    //clean up

    TENSOR ref_reqnt_out =
    ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_0.idx"));
  TENSOR ref_reqnt_out_min =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_1.idx"));
  TENSOR ref_reqnt_out_max =
    ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_eightbit_requantize_2.idx"));
   S_TENSOR ref_reqnt = ref_reqnt_out.lock();
   S_TENSOR ref_reqnt_min = ref_reqnt_out_min.lock();
   S_TENSOR ref_reqnt_max = ref_reqnt_out_max.lock();
/*
    temp_result = (meanPercentErr<unsigned char>(ref_reqnt.get(), out_reqnt.get()) + meanPercentErr<float>(ref_reqnt_min.get(), out_reqnt_min.get()) + meanPercentErr<float>(ref_reqnt_max.get(), out_reqnt_max.get()));
    if(temp_result > 0) {
      DEBUG("Requantize failed\r\n");
      failed();
      return;
    } else {
      DEBUG("Requantize passed\r\n");
    }

    DEBUG("Requantize completed!\r\n");*/

    //output
    TENSOR deqnt_out = ctx.add(new RamTensor<float>(out_c.lock()->getShape()));
    S_TENSOR out_deqnt = deqnt_out.lock();
    ctx.push(new DequantizeOp(), {reqnt_out, reqnt_out_min, reqnt_out_max}, {deqnt_out});

    TENSOR ref_deqnt_out = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/import-MatMul_0.idx"));
    S_TENSOR ref_deqnt = ref_deqnt_out.lock();
    /*double temp = meanPercentErr<float>(ref_deqnt.get(), out_deqnt.get());
    if(temp > 0.0001) {
      printf("dequantize failed (%.6f)\r\n", temp);
      const float* ref_ptr = ref_deqnt.get()->read<float>(0, 0);
      const float* test_ptr = out_deqnt.get()->read<float>(0, 0);
      for(uint32_t i; i < ref_deqnt->getSize(); i++) {
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
    }*/

    DEBUG("dequantize completed!\r\n");

    //input
    TENSOR bias = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-Variable_1_0.idx"));
    //output
    TENSOR output_z = ctx.add(new RamTensor<float>(deqnt_out.lock()->getShape())); 
    S_TENSOR out_z = output_z.lock();
    ctx.push(new AddOp<float, float>(), {deqnt_out, bias}, {output_z});
    ctx.eval();

    DEBUG("Add completed!\r\n");

    timer_stop();

    //load reference
    TENSOR ref_z = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntDeqntLayerZ/out/import-add_0.idx"));
    S_TENSOR ref_z_v = ref_z.lock();

    double result = meanPercentErr<float>(ref_z_v.get(), out_z.get());

    passed(result < 0.0001);

  }

  void runQntRelu() {

    testStart("runQntRelu");

    TENSOR input_z = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/in/import-add_0.idx"));
    TENSOR reshape_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reshape_dims_0.idx"));
    TENSOR reshape_out = ctx.add(new RamTensor<float>());

    timer_start();

    ctx.push(new ReshapeOp(), {input_z, reshape_dim}, {reshape_out});

    //min
    //input
    TENSOR min_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_min.idx"));
    //output
    TENSOR min_out = ctx.add(new RamTensor<float>({1}));
    ctx.push(new MinOp(), {reshape_out, min_reduce_dim}, {min_out});

    //max
    //input
    TENSOR max_reduce_dim = ctx.add(t_import.int_import("/fs/testData/mlpTest/runQntRelu/in/import-Relu_eightbit_reduction_dims_0_max.idx"));
    //output
    TENSOR max_out = ctx.add(new RamTensor<float>({1}));
    ctx.push(new MaxOp(), {reshape_out, max_reduce_dim}, {max_out});

    //quantization
    //output
    TENSOR qnt_out = ctx.add(new RamTensor<unsigned char>());
    TENSOR qnt_min = ctx.add(new RamTensor<float>({1}));
    TENSOR qnt_max = ctx.add(new RamTensor<float>({1}));
    ctx.push(new QuantizeV2Op(), {reshape_out, min_out, max_out}, {qnt_out, qnt_min, qnt_max});
    
    TENSOR out = ctx.add(new RamTensor<unsigned char>());
    TENSOR out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR out_max = ctx.add(new RamTensor<float>({1}));

    S_TENSOR out_val = out.lock();
    S_TENSOR out_min_val = out_min.lock();
    S_TENSOR out_max_val = out_max.lock();
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), {qnt_out, qnt_min, qnt_max}, {out, out_min, out_max});
    ctx.eval();

    timer_stop();

    TENSOR ref_out =
      ctx.add(t_import.ubyte_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_0.idx"));
    TENSOR ref_out_min = ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_1.idx"));
    TENSOR ref_out_max =
      ctx.add(t_import.float_import("/fs/testData/mlpTest/runQntRelu/out/import-Relu_eightbit_quantized_2.idx"));

    S_TENSOR ref_val = ref_out.lock();
    S_TENSOR ref_min_val = ref_out_min.lock();
    S_TENSOR ref_max_val = ref_out_max.lock();
    double result = meanPercentErr<unsigned char>(ref_val.get(), out_val.get());
    result += meanPercentErr<float>(ref_min_val.get(), out_min_val.get());
    result += meanPercentErr<float>(ref_max_val.get(), out_max_val.get());
    

    passed(result == 0);
  }


  void runAll() {
    runQuantization();
    runQntDeqntLayerZ();
    runQntRelu();
  }
};

#endif  //UTENSOR_MLP_TEST
