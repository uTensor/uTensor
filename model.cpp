//FIXME: include order
#include "MatrixOps.hpp"
#include "tensorIdxImporter.hpp"
#include "model.hpp"
#include "ArrayOps.hpp"
#include "context.hpp"
#include "MathOps.hpp"
#include "NnOps.hpp"
#include "tensor.hpp"



void get_quantized_graph_ctx(Context& ctx, Tensor* input_0) {

{ // add tensor for placeholders
    
    ctx.add(input_0, "Placeholder:0");
    
}



{
    TensorIdxImporter t_import;
    ctx.add(t_import.ubyte_import("/fs/idx_data/Variable_quint8_const_0.idx"), "Variable_quint8_const:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_min_0.idx"), "Variable_min:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_max_0.idx"), "Variable_max:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/MatMul_eightbit_reshape_dims_0.idx"), "MatMul_eightbit_reshape_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/MatMul_eightbit_reduction_dims_0.idx"), "MatMul_eightbit_reduction_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_1_0.idx"), "Variable_1:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/Relu_eightbit_reshape_dims_0.idx"), "Relu_eightbit_reshape_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/Relu_eightbit_reduction_dims_0.idx"), "Relu_eightbit_reduction_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.ubyte_import("/fs/idx_data/Variable_2_quint8_const_0.idx"), "Variable_2_quint8_const:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_2_min_0.idx"), "Variable_2_min:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_2_max_0.idx"), "Variable_2_max:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_3_0.idx"), "Variable_3:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/Relu_1_eightbit_reshape_dims_0.idx"), "Relu_1_eightbit_reshape_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/Relu_1_eightbit_reduction_dims_0.idx"), "Relu_1_eightbit_reduction_dims:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.ubyte_import("/fs/idx_data/Variable_4_quint8_const_0.idx"), "Variable_4_quint8_const:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_4_min_0.idx"), "Variable_4_min:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_4_max_0.idx"), "Variable_4_max:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.float_import("/fs/idx_data/Variable_5_0.idx"), "Variable_5:0", 1);
    
}


{
    TensorIdxImporter t_import;
    ctx.add(t_import.int_import("/fs/idx_data/y_pred_dimension_0.idx"), "y_pred/dimension:0", 1);
    
}

{
    ctx.add(new RamTensor<float>(), "MatMul_eightbit_reshape_Placeholder:0", 2);
    ctx.push(new ReshapeOp(), 
             { "Placeholder:0", "MatMul_eightbit_reshape_dims:0" },
             { "MatMul_eightbit_reshape_Placeholder:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "MatMul_eightbit_max_Placeholder:0", 1);
    
    ctx.push(new MaxOp(), 
             { "MatMul_eightbit_reshape_Placeholder:0", "MatMul_eightbit_reduction_dims:0" },
             { "MatMul_eightbit_max_Placeholder:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "MatMul_eightbit_min_Placeholder:0", 1);
    
    ctx.push(new MinOp(), 
             { "MatMul_eightbit_reshape_Placeholder:0", "MatMul_eightbit_reduction_dims:0" },
             { "MatMul_eightbit_min_Placeholder:0" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "MatMul_eightbit_quantize_Placeholder:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_quantize_Placeholder:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_quantize_Placeholder:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "Placeholder:0",  "MatMul_eightbit_min_Placeholder:0", "MatMul_eightbit_max_Placeholder:0" },
             {  "MatMul_eightbit_quantize_Placeholder:0",  "MatMul_eightbit_quantize_Placeholder:1", "MatMul_eightbit_quantize_Placeholder:2" });
}

{
    ctx.add(new RamTensor<int>(), "MatMul_eightbit_quantized_mat_mul:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_quantized_mat_mul:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_quantized_mat_mul:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "MatMul_eightbit_quantize_Placeholder:0", "MatMul_eightbit_quantize_Placeholder:1", "MatMul_eightbit_quantize_Placeholder:2", "Variable_quint8_const:0", "Variable_min:0",  "Variable_max:0" },
             { "MatMul_eightbit_quantized_mat_mul:0", "MatMul_eightbit_quantized_mat_mul:1",  "MatMul_eightbit_quantized_mat_mul:2" });
}

{
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul_eightbit_quantized_mat_mul:0", "MatMul_eightbit_quantized_mat_mul:1", "MatMul_eightbit_quantized_mat_mul:2" },
             { "MatMul_eightbit_requant_range:0", "MatMul_eightbit_requant_range:1" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "MatMul_eightbit_requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_eightbit_requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul_eightbit_quantized_mat_mul:0", "MatMul_eightbit_quantized_mat_mul:1", "MatMul_eightbit_quantized_mat_mul:2", "MatMul_eightbit_requant_range:0", "MatMul_eightbit_requant_range:1" },
             { "MatMul_eightbit_requantize:0", "MatMul_eightbit_requantize:1", "MatMul_eightbit_requantize:2" });
}

{
    ctx.add(new RamTensor<float>(), "MatMul:0", 1);
    ctx.push(new DequantizeOp(), 
             { "MatMul_eightbit_requantize:0", "MatMul_eightbit_requantize:1", "MatMul_eightbit_requantize:2" },
             { "MatMul:0" });
}

{
    ctx.add(new RamTensor<float>(), "add:0", 2);
    ctx.push(new AddOp<float, float>(),
             { "MatMul:0", "Variable_1:0" }, 
             { "add:0" });
}

{
    ctx.add(new RamTensor<float>(), "Relu_eightbit_reshape_add:0", 2);
    ctx.push(new ReshapeOp(), 
             { "add:0", "Relu_eightbit_reshape_dims:0" },
             { "Relu_eightbit_reshape_add:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "Relu_eightbit_min_add:0", 1);
    
    ctx.push(new MinOp(), 
             { "Relu_eightbit_reshape_add:0", "Relu_eightbit_reduction_dims:0" },
             { "Relu_eightbit_min_add:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "Relu_eightbit_max_add:0", 1);
    
    ctx.push(new MaxOp(), 
             { "Relu_eightbit_reshape_add:0", "Relu_eightbit_reduction_dims:0" },
             { "Relu_eightbit_max_add:0" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "Relu_eightbit_quantize_add:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_eightbit_quantize_add:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_eightbit_quantize_add:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "add:0",  "Relu_eightbit_min_add:0", "Relu_eightbit_max_add:0" },
             {  "Relu_eightbit_quantize_add:0",  "Relu_eightbit_quantize_add:1", "Relu_eightbit_quantize_add:2" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "Relu_eightbit_quantized:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_eightbit_quantized:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_eightbit_quantized:2", 1);
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), 
             { "Relu_eightbit_quantize_add:0", "Relu_eightbit_quantize_add:1", "Relu_eightbit_quantize_add:2" },
             { "Relu_eightbit_quantized:0", "Relu_eightbit_quantized:1", "Relu_eightbit_quantized:2" });
}

{
    ctx.add(new RamTensor<int>(), "MatMul_1_eightbit_quantized_mat_mul:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_quantized_mat_mul:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_quantized_mat_mul:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "Relu_eightbit_quantized:0", "Relu_eightbit_quantized:1", "Relu_eightbit_quantized:2", "Variable_2_quint8_const:0", "Variable_2_min:0",  "Variable_2_max:0" },
             { "MatMul_1_eightbit_quantized_mat_mul:0", "MatMul_1_eightbit_quantized_mat_mul:1",  "MatMul_1_eightbit_quantized_mat_mul:2" });
}

{
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul_1_eightbit_quantized_mat_mul:0", "MatMul_1_eightbit_quantized_mat_mul:1", "MatMul_1_eightbit_quantized_mat_mul:2" },
             { "MatMul_1_eightbit_requant_range:0", "MatMul_1_eightbit_requant_range:1" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "MatMul_1_eightbit_requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_1_eightbit_requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul_1_eightbit_quantized_mat_mul:0", "MatMul_1_eightbit_quantized_mat_mul:1", "MatMul_1_eightbit_quantized_mat_mul:2", "MatMul_1_eightbit_requant_range:0", "MatMul_1_eightbit_requant_range:1" },
             { "MatMul_1_eightbit_requantize:0", "MatMul_1_eightbit_requantize:1", "MatMul_1_eightbit_requantize:2" });
}

{
    ctx.add(new RamTensor<float>(), "MatMul_1:0", 1);
    ctx.push(new DequantizeOp(), 
             { "MatMul_1_eightbit_requantize:0", "MatMul_1_eightbit_requantize:1", "MatMul_1_eightbit_requantize:2" },
             { "MatMul_1:0" });
}

{
    ctx.add(new RamTensor<float>(), "add_1:0", 2);
    ctx.push(new AddOp<float, float>(),
             { "MatMul_1:0", "Variable_3:0" }, 
             { "add_1:0" });
}

{
    ctx.add(new RamTensor<float>(), "Relu_1_eightbit_reshape_add_1:0", 2);
    ctx.push(new ReshapeOp(), 
             { "add_1:0", "Relu_1_eightbit_reshape_dims:0" },
             { "Relu_1_eightbit_reshape_add_1:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "Relu_1_eightbit_min_add_1:0", 1);
    
    ctx.push(new MinOp(), 
             { "Relu_1_eightbit_reshape_add_1:0", "Relu_1_eightbit_reduction_dims:0" },
             { "Relu_1_eightbit_min_add_1:0" });
}

{   
    ctx.add(new RamTensor<float>({ 1 }), "Relu_1_eightbit_max_add_1:0", 1);
    
    ctx.push(new MaxOp(), 
             { "Relu_1_eightbit_reshape_add_1:0", "Relu_1_eightbit_reduction_dims:0" },
             { "Relu_1_eightbit_max_add_1:0" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "Relu_1_eightbit_quantize_add_1:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1_eightbit_quantize_add_1:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1_eightbit_quantize_add_1:2", 1);
    ctx.push(new QuantizeV2Op(),
             {  "add_1:0",  "Relu_1_eightbit_min_add_1:0", "Relu_1_eightbit_max_add_1:0" },
             {  "Relu_1_eightbit_quantize_add_1:0",  "Relu_1_eightbit_quantize_add_1:1", "Relu_1_eightbit_quantize_add_1:2" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "Relu_1_eightbit_quantized:0", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1_eightbit_quantized:1", 1);
    ctx.add(new RamTensor<float>({1}), "Relu_1_eightbit_quantized:2", 1);
    ctx.push(new ReluOp<uint8_t, float, uint8_t>(), 
             { "Relu_1_eightbit_quantize_add_1:0", "Relu_1_eightbit_quantize_add_1:1", "Relu_1_eightbit_quantize_add_1:2" },
             { "Relu_1_eightbit_quantized:0", "Relu_1_eightbit_quantized:1", "Relu_1_eightbit_quantized:2" });
}

{
    ctx.add(new RamTensor<int>(), "MatMul_2_eightbit_quantized_mat_mul:0", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_quantized_mat_mul:1", 2);
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_quantized_mat_mul:2", 2);
    ctx.push(new QntMatMulOp<uint8_t, uint8_t, int>(), 
             { "Relu_1_eightbit_quantized:0", "Relu_1_eightbit_quantized:1", "Relu_1_eightbit_quantized:2", "Variable_4_quint8_const:0", "Variable_4_min:0",  "Variable_4_max:0" },
             { "MatMul_2_eightbit_quantized_mat_mul:0", "MatMul_2_eightbit_quantized_mat_mul:1",  "MatMul_2_eightbit_quantized_mat_mul:2" });
}

{
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_requant_range:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_requant_range:1", 1);
    ctx.push(new Requantization_RangeOp(),
             { "MatMul_2_eightbit_quantized_mat_mul:0", "MatMul_2_eightbit_quantized_mat_mul:1", "MatMul_2_eightbit_quantized_mat_mul:2" },
             { "MatMul_2_eightbit_requant_range:0", "MatMul_2_eightbit_requant_range:1" });
}

{
    ctx.add(new RamTensor<uint8_t>(), "MatMul_2_eightbit_requantize:0", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_requantize:1", 1);
    ctx.add(new RamTensor<float>({1}), "MatMul_2_eightbit_requantize:2", 1);
    ctx.push(new RequantizeOp(),
             { "MatMul_2_eightbit_quantized_mat_mul:0", "MatMul_2_eightbit_quantized_mat_mul:1", "MatMul_2_eightbit_quantized_mat_mul:2", "MatMul_2_eightbit_requant_range:0", "MatMul_2_eightbit_requant_range:1" },
             { "MatMul_2_eightbit_requantize:0", "MatMul_2_eightbit_requantize:1", "MatMul_2_eightbit_requantize:2" });
}

{
    ctx.add(new RamTensor<float>(), "MatMul_2:0", 1);
    ctx.push(new DequantizeOp(), 
             { "MatMul_2_eightbit_requantize:0", "MatMul_2_eightbit_requantize:1", "MatMul_2_eightbit_requantize:2" },
             { "MatMul_2:0" });
}

{
    ctx.add(new RamTensor<float>(), "add_2:0", 1);
    ctx.push(new AddOp<float, float>(),
             { "MatMul_2:0", "Variable_5:0" }, 
             { "add_2:0" });
}

{
    ctx.add(new RamTensor<int>(), "y_pred:0", 0);
    ctx.push(new ArgMaxOp<float, int>(), 
             { "add_2:0", "y_pred/dimension:0" },
             { "y_pred:0" });
}

}
