#include "deep_mnist_mlp.hpp"
#include "uTensor/util/math_utils.hpp"

void tensorQuantize(Context& ctx, TName input, TName output,
  TName out_min, TName out_max) {

    //reshape
    S_TENSOR reduce_dim = ctx.add(new RamTensor<int>({1}), "reduce_dim");
    *(reduce_dim->write<int>(0, 0)) = 0;

    ctx.add(new RamTensor<float>(), "reshape_out");

    S_TENSOR reshape_shape = ctx.add(new RamTensor<int>({2}), "reshape_shape");
    auto inputSize = ctx.get(input)->getSize();
    reshape_shape->write<int>(0, 0)[0] = inputSize;
    reshape_shape->write<int>(0, 0)[1] = -1;

    ctx.push(new ReshapeOp(), {input, "reshape_shape"}, {"reshape_out"});


    //Min and Max of (reshaped) input
    ctx.add(new RamTensor<float>({1}), "min_out");
    ctx.add(new RamTensor<float>({1}), "max_out");
    ctx.push(new MinOp(), {"reshape_out", "reduce_dim"}, {"min_out"});
    ctx.push(new MaxOp(), {"reshape_out", "reduce_dim"}, {"max_out"});

    ctx.push(new QuantizeV2Op(), {input, "min_out", "max_out"}, {output, out_min, out_max});

}

void ReluLayer(Context& ctx, TName x, TName x_min, TName x_max,
   TName w, TName w_min, TName w_max, TName b,
    TName z_output) {
  
    //quantized matmul

    S_TENSOR out_c = ctx.add(new RamTensor<int>(), "out_c");

    ctx.add(new RamTensor<float>({1}), "matmul_out_min");
    ctx.add(new RamTensor<float>({1}), "matmul_out_max");

    // printf("QntMulOp TensorShape:");
    // printf("\r\nx :");
    // printVector(ctx.get(x)->getShape());
    // printf("\r\nw :");
    // printVector(ctx.get(w)->getShape());
    // printf("\r\n");

    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), {x, x_min, x_max, w, w_min, w_max}, {"out_c", "matmul_out_min", "matmul_out_max"});

    //Requantization_Range
    S_TENSOR req_out_min = ctx.add(new RamTensor<float>({1}), "req_out_min");
    S_TENSOR req_out_max = ctx.add(new RamTensor<float>({1}), "req_out_max");
    ctx.push(new Requantization_RangeOp(), {"out_c", "matmul_out_min", "matmul_out_max"}, {"req_out_min", "req_out_max"});

    //Requantize
    ctx.add(new RamTensor<unsigned char>(), "reqnt_out");
    ctx.add(new RamTensor<float>({1}), "reqnt_out_min");
    ctx.add(new RamTensor<float>({1}), "reqnt_out_max");
    ctx.push(new RequantizeOp(), {"out_c", "matmul_out_min", "matmul_out_max", "req_out_min", "req_out_max"}, {"reqnt_out", "reqnt_out_min", "reqnt_out_max"});

    TensorShape out_shape = out_c->getShape();
    //clean up

    S_TENSOR deqnt_out = ctx.add(new RamTensor<float>(), "deqnt_out");
    ctx.push(new DequantizeOp(), {"reqnt_out", "reqnt_out_min", "reqnt_out_max"}, {"deqnt_out"});

    ctx.push(new AddOp<float, float>(), {"deqnt_out", b}, {z_output});

}

void PredLayer(Context &ctx, TName input, TName input_min,
               TName input_max, TName output, TName w, TName w_min, TName w_max, TName bias, TName dim) {

  S_TENSOR out_mat_pred = ctx.add(new RamTensor<int>(), "out_mat_pred");
  S_TENSOR matmul_out_min_pred = ctx.add(new RamTensor<float>({1}), "matmul_out_min_pred");
  S_TENSOR matmul_out_max_pred = ctx.add(new RamTensor<float>({1}), "matmul_out_max_pred");

  //MatMul
  ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), {input, input_min, input_max, w, w_min, w_max},
          {"out_mat_pred", "matmul_out_min_pred", "matmul_out_max_pred"});

  //Requantization_Range
  ctx.add(new RamTensor<float>({1}), "req_out_min_pred");
  ctx.add(new RamTensor<float>({1}), "req_out_max_pred");
  ctx.push(new Requantization_RangeOp(), {"out_mat_pred", "matmul_out_min_pred", "matmul_out_max_pred"},
          {"req_out_min_pred", "req_out_max_pred"});

  //Requantize
  S_TENSOR reqnt_out = ctx.add(new RamTensor<unsigned char>(), "reqnt_out_pred");
  S_TENSOR reqnt_out_min = ctx.add(new RamTensor<float>({1}), "reqnt_out_min_pred");
  S_TENSOR reqnt_out_max = ctx.add(new RamTensor<float>({1}), "reqnt_out_max_pred");
  ctx.push(new RequantizeOp(), {"out_mat_pred", "matmul_out_min_pred", "matmul_out_max_pred", "req_out_min_pred", "req_out_max_pred"},
          {"reqnt_out_pred", "reqnt_out_min_pred", "reqnt_out_max_pred"});

  //dequantize
  ctx.add(new RamTensor<float>(), "deqnt_out_pred");
  ctx.push(new DequantizeOp(), {"reqnt_out_pred", "reqnt_out_min_pred", "reqnt_out_max_pred"}, {"deqnt_out_pred"});

  //Add
  ctx.add(new RamTensor<float>(), "output_z_pred"); 
  ctx.push(new AddOp<float, float>(), {"deqnt_out_pred", bias}, {"output_z_pred"});

  //ArgMax
  ctx.push(new ArgMaxOp<float, int>(), {"output_z_pred", dim}, {output});
}

int runMLP(const char* inputIdxFile) {
  TensorIdxImporter t_import;
  Context ctx;
  ctx.add(new RamTensor<unsigned char>(), "x_quantized"); 
  ctx.add(new RamTensor<float>({1}), "x_min");
  ctx.add(new RamTensor<float>({1}), "x_max"); 
  ctx.add(t_import.float_import(inputIdxFile), "x");

  tensorQuantize(ctx, "x", "x_quantized", "x_min", "x_max");
  ctx.eval();

 //relu layer first

  ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_quint8_const_0.idx"), "w");
  ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_min_0.idx"), "w_min");
  ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_max_0.idx"), "w_max");
  ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_1_0.idx"), "b");
  ctx.add(new RamTensor<unsigned char>(), "relu_output");
  ctx.add(new RamTensor<float>({1}), "relu_min");
  ctx.add(new RamTensor<float>({1}), "relu_max");
  ctx.add(new RamTensor<float>(), "z_output"); 

  ReluLayer(ctx, "x_quantized", "x_min", "x_max", "w", "w_min", "w_max", "b", "z_output");
  ctx.eval();

  ctx.add(new RamTensor<unsigned char>(), "z_qnt_output");
  ctx.add(new RamTensor<float>({1}), "z_min");
  ctx.add(new RamTensor<float>({1}), "z_max");
  tensorQuantize(ctx, "z_output", "z_qnt_output", "z_min", "z_max");

  ctx.push(new ReluOp<unsigned char, float, unsigned char>(), {"z_qnt_output", "z_min", "z_max"}, {"relu_output", "relu_min", "relu_max"});

  ctx.eval();

  //relu layer 2
  ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/import-Variable_2_quint8_const_0.idx"), "w2");
  ctx.add(t_import.float_import(
     "/fs/testData/deep_mlp/import-Variable_2_min_0.idx"), "w_min2");
  ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/import-Variable_2_max_0.idx"), "w_max2");
  ctx.add(t_import.float_import("/fs/testData/deep_mlp/import-Variable_3_0.idx"), "b2");
  ctx.add(new RamTensor<unsigned char>(), "relu_output2");
  ctx.add(new RamTensor<float>({1}), "relu_min2");
  ctx.add(new RamTensor<float>({1}), "relu_max2");

  ctx.add(new RamTensor<float>(), "z_output2"); 
  ReluLayer(ctx, "relu_output", "relu_min", "relu_max", "w2", "w_min2", "w_max2", "b2", "z_output2");
  ctx.eval();

  ctx.add(new RamTensor<unsigned char>(), "z_qnt_output2");
  ctx.add(new RamTensor<float>({1}), "z_min2");
  ctx.add(new RamTensor<float>({1}), "z_max2");
  tensorQuantize(ctx, "z_output2", "z_qnt_output2", "z_min2", "z_max2");

  ctx.push(new ReluOp<unsigned char, float, unsigned char>(), {"z_qnt_output2", "z_min2", "z_max2"}, {"relu_output2", "relu_min2", "relu_max2"});

  ctx.eval();

  ctx.add(t_import.ubyte_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_quint8_const_0.idx"), "w3");
  ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_min_0.idx"), "w2_min");
  ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/MatMul_2_eightbit_quantized_mat_mul/"
      "inputs/Variable_4_max_0.idx"), "w2_max");
  ctx.add(t_import.float_import(
      "/fs/testData/deep_mlp/runPredLayer/add_2/inputs/Variable_5_0.idx"), "bias2");
  ctx.add(t_import.int_import(
      "/fs/testData/deep_mlp/runPredLayer/y_pred/inputs/"
      "y_pred-dimension_0.idx"), "dim");

  S_TENSOR pred = ctx.add(new RamTensor<int>(), "pred");
  PredLayer(ctx, "relu_output2", "relu_min2", "relu_max2", "pred", "w3", "w2_min", "w2_max", "bias2", "dim");
  ctx.eval();


  Tensor* ref_out = t_import.float_import(
    "/fs/testData/deep_mlp/runPredLayer/y_pred/outputs/y_pred_0.idx");
  Tensor* ref_pred = TensorCast<float, int>(ref_out);

  double result = utils::meanPercentErr<int>(ref_pred, pred.get());
  
  if (result < 0.0001) {
    printf("PASSED %.8f\r\n\r\n", result);
  } else {
    printf("FAILED %.8f\r\n\r\n", result);
  }

  return *(pred->read<int>(0, 0));
  // output layer
}
