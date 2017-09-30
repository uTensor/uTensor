# Owners

Neil Tan, (more)

# Description

Python version of a quantize->matmul->dequantize pipeline

Used for:

- Precision Study and Characterize
- Debug data generation
- Ops covered: quantize_v2, quantized_mat_mul, requantization_range, requantize, dequantize

# Useful Commands

    ipython3
    run uint8uint8int32.py
    import [node viewer] as nv
    nv.init(tf.get_default_graph)
    nv.ls()
    nv.snap("qMatMul")
    

# TODO
TBD

# Reference

- quantize_v2:
    gen_array_ops.py:2162
    quantize_op.cc:64

- quantized_mat_mul:
    - gen_math_ops.py:1619
    - quantized_matmul_ops.cc:178
    - reference_gemm.h:29
    - quantization_utils.h:121, 112