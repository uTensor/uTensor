#include "mbed.h"
#include "tensor.hpp"
#include "test.hpp"
#include "tensorIdxImporter.hpp"
#include "MathOps.hpp"
#include "MatrixOps.hpp"
#include "NnOps.hpp"
#include "ArrayOps.hpp"
#include "uTensor_util.hpp"
#include "uTensorBase.hpp"
#include "context.hpp"

void tensorQuantize(Context& ctx, TENSOR input, TENSOR output,
  TENSOR out_min, TENSOR out_max) {

    //reshape
    TENSOR reduce_dim = ctx.add(new RamTensor<int>({1}));
    TENSOR reshape_out = ctx.add(new RamTensor<float>());

    TENSOR reshape_shape = ctx.add(new RamTensor<int>());

    *(reduce_dim.lock()->write<int>(0, 0)) = 0;
    ctx.push(new ReshapeOp(), {input, reshape_shape}, {reshape_out});


    //Min and Max of (reshaped) input
    TENSOR min_out = ctx.add(new RamTensor<float>({1}));
    TENSOR max_out = ctx.add(new RamTensor<float>({1}));
    ctx.push(new MinOp(), {reshape_out, reduce_dim}, {min_out});
    ctx.push(new MaxOp(), {reshape_out, reduce_dim}, {max_out});

    ctx.push(new QuantizeV2Op(), {reshape_out, min_out, max_out}, {output, out_min, out_max});
}

void ReluLayer(Context& ctx, TENSOR x, TENSOR x_min, TENSOR x_max,
   TENSOR w, TENSOR w_min, TENSOR w_max, TENSOR b,
    TENSOR output, TENSOR output_min, TENSOR output_max) {
  
    //quantized matmul

    TENSOR out_c = ctx.add(new RamTensor<int>());

    TENSOR matmul_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR matmul_out_max = ctx.add(new RamTensor<float>({1}));

    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), {x, x_min, x_max, w, w_min, w_max}, {out_c, matmul_out_min, matmul_out_max});

    //Requantization_Range
    TENSOR req_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR req_out_max = ctx.add(new RamTensor<float>({1}));
    ctx.push(new Requantization_RangeOp(), {out_c, matmul_out_min, matmul_out_max}, {req_out_min, req_out_max});

    //Requantize
    TENSOR reqnt_out = ctx.add(new RamTensor<unsigned char>());
    TENSOR reqnt_out_min = ctx.add(new RamTensor<float>({1}));
    TENSOR reqnt_out_max = ctx.add(new RamTensor<float>({1}));
    ctx.push(new RequantizeOp(), {out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max}, {reqnt_out, reqnt_out_min, reqnt_out_max});

    Shape out_shape = out_c.lock()->getShape();
    //clean up

    TENSOR deqnt_out = ctx.add(new RamTensor<float>());
    ctx.push(new DequantizeOp(), {reqnt_out, reqnt_out_min, reqnt_out_max}, {deqnt_out});
    TENSOR z_output = ctx.add(new RamTensor<float>()); 

    ctx.push(new AddOp<float, float>(), {deqnt_out, b}, {z_output});

    TENSOR z_qnt_output = ctx.add(new RamTensor<unsigned char>());
    TENSOR z_min = ctx.add(new RamTensor<float>({1}));
    TENSOR z_max = ctx.add(new RamTensor<float>({1}));
    tensorQuantize(ctx, z_output, z_qnt_output, z_min, z_max);

    ctx.push(new ReluOp<unsigned char, float, unsigned char>(), {z_qnt_output, z_min, z_max}, {output, output_min, output_max});
}

void PredLayer(Context &ctx, TENSOR input, TENSOR input_min,
               TENSOR input_max, TENSOR output, TENSOR w, TENSOR w_min, TENSOR w_max, TENSOR bias, TENSOR dim) {

  TENSOR out_c = ctx.add(new RamTensor<int>());
  TENSOR matmul_out_min = ctx.add(new RamTensor<float>({1}));
  TENSOR matmul_out_max = ctx.add(new RamTensor<float>({1}));

  //MatMul
  ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), {input, input_min, input_max, w, w_min, w_max},
          {out_c, matmul_out_min, matmul_out_max});

  //Requantization_Range
  TENSOR req_out_min = ctx.add(new RamTensor<float>({1}));
  TENSOR req_out_max = ctx.add(new RamTensor<float>({1}));
  ctx.push(new Requantization_RangeOp(), {out_c, matmul_out_min, matmul_out_max},
          {req_out_min, req_out_max});

  //Requantize
  TENSOR reqnt_out = ctx.add(new RamTensor<unsigned char>());
  TENSOR reqnt_out_min = ctx.add(new RamTensor<float>({1}));
  TENSOR reqnt_out_max = ctx.add(new RamTensor<float>({1}));
  ctx.push(new RequantizeOp(), {out_c, matmul_out_min, matmul_out_max, req_out_min, req_out_max},
          {reqnt_out, reqnt_out_min, reqnt_out_max});

  //dequantize
  TENSOR deqnt_out = ctx.add(new RamTensor<float>());
  ctx.push(new DequantizeOp(), {reqnt_out, reqnt_out_min, reqnt_out_max}, {deqnt_out});

  //Add
  TENSOR output_z = ctx.add(new RamTensor<float>()); 
  ctx.push(new AddOp<float, float>(), {deqnt_out, bias}, {output_z});

  //ArgMax
  ctx.push(new ArgMaxOp<float, int>(), {output_z, dim}, {output});
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


int runMLP(string inputIdxFile) {
  TensorIdxImporter t_import;
  Context ctx;
  TENSOR x_quantized = ctx.add(new RamTensor<unsigned char>()); 
  TENSOR x_min = ctx.add(new RamTensor<float>({1}));
  TENSOR x_max = ctx.add(new RamTensor<float>({1})); 
  TENSOR x = ctx.add(t_import.float_import(inputIdxFile));
  S_TENSOR xs_quantized = x_quantized.lock();
  S_TENSOR xs_min = x_min.lock();
  S_TENSOR xs_max = x_max.lock(); 

  tensorQuantize(ctx, x, x_quantized, x_min, x_max);
  ctx.eval();

 //relu layer first

  TENSOR w = ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_quint8_const_0.idx"));
  TENSOR w_min =
      ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_min_0.idx"));
  TENSOR w_max =
      ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_max_0.idx"));
  TENSOR b =
      ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_1_0.idx"));
  TENSOR relu_output = ctx.add(new RamTensor<unsigned char>());
  TENSOR relu_min = ctx.add(new RamTensor<float>({1}));
  TENSOR relu_max = ctx.add(new RamTensor<float>({1}));
  S_TENSOR relus_output = relu_output.lock(); 
  S_TENSOR relus_min = relu_min.lock();
  S_TENSOR relus_max = relu_max.lock();


  ReluLayer(ctx, x_quantized, x_min, x_max, w, w_min, w_max, b, relu_output,
          relu_min, relu_max);
  ctx.eval();
  //relu layer 2
  TENSOR w2 = ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_2_quint8_const_0.idx"));
  TENSOR w_min2 = ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_min_0.idx"));
  TENSOR w_max2 = ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_max_0.idx"));
  TENSOR b2 = ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_3_0.idx"));
  TENSOR relu_output2 = ctx.add(new RamTensor<unsigned char>());
  TENSOR relu_min2 = ctx.add(new RamTensor<float>({1}));
  TENSOR relu_max2 = ctx.add(new RamTensor<float>({1}));

  S_TENSOR relus_output2 = relu_output2.lock(); 
  S_TENSOR relus_min2 = relu_min2.lock();
  S_TENSOR relus_max2 = relu_max2.lock();

  ReluLayer(ctx, relu_output, relu_min, relu_max, w2, w_min2, w_max2, b2, relu_output2,
            relu_min2, relu_max2);
  ctx.eval();

  TENSOR w3 = ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_quint8_const_0.idx"));
  TENSOR w2_min = ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_min_0.idx"));
  TENSOR w2_max = ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_max_0.idx"));
  TENSOR bias2 = ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/add_2/inputs/Variable_5_0.idx"));
  TENSOR dim = ctx.add(t_import.int_import(
      "/fs/testData/deep_mlp/runPredLayer/y_pred/inputs/"
      "y_pred-dimension_0.idx"));

  TENSOR pred = ctx.add(new RamTensor<int>());
  PredLayer(ctx, relu_output2, relu_min2, relu_max2, pred, w3, w2_min, w2_max, bias2, dim);
  S_TENSOR pred_val = pred.lock();
  ctx.eval();


  Tensor* ref_out = t_import.float_import(
    "/fs/testData/deep_mlp/runPredLayer/y_pred/outputs/y_pred_0.idx");
  Tensor* ref_pred = TensorCast<float, int>(ref_out);

  double result = Test::meanPercentErr<int>(ref_pred, pred_val.get());
  
  if (result < 0.0001) {
    printf("PASSED %.8f\r\n\r\n", result);
  } else {
    printf("FAILED %.8f\r\n\r\n", result);
  }

  return *(pred.lock()->read<int>(0, 0));
  // output layer
}
